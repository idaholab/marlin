/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/
#include "PetscFDHierarchy.h"
#include "DomainAction.h"
#include "TensorRobinBC.h"
#include "MooseError.h"

#include "libmesh/int_range.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>

namespace PetscFD
{

namespace
{
/// minimum coarse extent (cells per axis)
constexpr int64_t coarsest_extent = 8;

/// Marlin axis of DMDA axis a
unsigned int
marlinAxis(unsigned int dim, unsigned int a)
{
  return dim - 1 - a;
}

struct DMDAInfo
{
  PetscInt dim;
  /// global extents, ranks per axis, owned corner and size (DMDA axis order)
  std::array<PetscInt, 3> M{{1, 1, 1}}, procs{{1, 1, 1}}, start{{0, 0, 0}}, size{{1, 1, 1}};
};

DMDAInfo
getInfo(DM da)
{
  const MPI_Comm comm = PetscObjectComm((PetscObject)da);
  DMDAInfo info;
  PetscCallAbort(comm,
                 DMDAGetInfo(da,
                             &info.dim,
                             &info.M[0],
                             &info.M[1],
                             &info.M[2],
                             &info.procs[0],
                             &info.procs[1],
                             &info.procs[2],
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr));
  PetscCallAbort(comm,
                 DMDAGetCorners(da,
                                &info.start[0],
                                &info.start[1],
                                &info.start[2],
                                &info.size[0],
                                &info.size[1],
                                &info.size[2]));
  return info;
}

/// periodic flag of each DMDA axis
std::array<bool, 3>
periodicAxes(DM da)
{
  std::array<DMBoundaryType, 3> bt;
  PetscCallAbort(PetscObjectComm((PetscObject)da),
                 DMDAGetInfo(da,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             &bt[0],
                             &bt[1],
                             &bt[2],
                             nullptr));
  return {{bt[0] == DM_BOUNDARY_PERIODIC,
           bt[1] == DM_BOUNDARY_PERIODIC,
           bt[2] == DM_BOUNDARY_PERIODIC}};
}

/**
 * Fold the column index c (DMDA axis a, global extent M) back into the domain: unchanged
 * inside, a ghost in layer l is its mirror (owned cell l from the face) times the face's
 * factor at the spacing h (per Marlin axis, read only on an axis with faces). Returns the
 * factor (1 inside or periodic).
 */
Real
foldIndex(PetscInt & c,
          PetscInt M,
          bool periodic,
          const FaceGhosts & faces,
          unsigned int marlin_axis,
          const std::vector<Real> & h)
{
  if (periodic || (c >= 0 && c < M))
    return 1.0;
  const bool upper = c >= M;
  const auto layer = unsigned(upper ? c - M : -c - 1);
  c = upper ? M - 1 - PetscInt(layer) : PetscInt(layer);
  return faces.factor(2 * marlin_axis + upper, layer, h[marlin_axis]);
}

/// an entry's offset in DMDA axis order
std::array<PetscInt, 3>
dmdaOffset(const FDStencilEntry & entry, unsigned int dim)
{
  std::array<PetscInt, 3> offset{{0, 0, 0}};
  for (const auto a : make_range(dim))
    offset[a] = entry.offset[marlinAxis(dim, a)];
  return offset;
}

/// the constant-coefficient entries merged by offset, in DMDA axis order
std::map<std::array<PetscInt, 3>, PetscScalar>
mergedStencil(const std::vector<FDStencilEntry> & stencil, unsigned int dim, PetscReal scale)
{
  std::map<std::array<PetscInt, 3>, PetscScalar> merged;
  for (const auto & entry : stencil)
    if (!entry.field.defined())
      merged[dmdaOffset(entry, dim)] += scale * entry.coefficient;
  return merged;
}

/// an entry with a per-cell field, which is read on the host per row
struct FieldEntry
{
  std::array<PetscInt, 3> offset;
  PetscScalar value;
  torch::Tensor field;
};

/// the entries with a per-cell field (scaled), each field checked against the owned block
std::vector<FieldEntry>
fieldEntries(const std::vector<FDStencilEntry> & stencil, const DMDAInfo & info, PetscReal scale)
{
  const auto dim = unsigned(info.dim);
  PetscInt cells = 1;
  for (const auto a : make_range(dim))
    cells *= info.size[a];
  std::vector<FieldEntry> entries;
  for (const auto & entry : stencil)
    if (entry.field.defined())
    {
      if (entry.field.numel() != cells)
        mooseError("A stencil field has ", entry.field.numel(), " cells, the owned block ", cells);
      entries.push_back({dmdaOffset(entry, dim),
                         scale * entry.coefficient,
                         entry.field.to(torch::kCPU, torch::kFloat64).contiguous()});
    }
  return entries;
}

/// Call row_fn(row, cols, values) for every owned row of the folded operator on da
template <typename RowFn>
void
forEachFoldedRow(DM da,
                 const std::vector<FDStencilEntry> & stencil,
                 PetscReal scale,
                 const FaceGhosts & faces,
                 const std::vector<Real> & h,
                 RowFn row_fn)
{
  const auto info = getInfo(da);
  const auto dim = unsigned(info.dim);
  const auto periodic = periodicAxes(da);
  const auto merged = mergedStencil(stencil, dim, scale);
  const auto field_entries = fieldEntries(stencil, info, scale);

  std::vector<MatStencil> cols;
  std::vector<PetscScalar> values;
  // the owned block's cells in order: DMDA x fastest is the Marlin tensor's contiguous axis
  PetscInt cell = 0;
  for (PetscInt k = info.start[2]; k < info.start[2] + info.size[2]; ++k)
    for (PetscInt j = info.start[1]; j < info.start[1] + info.size[1]; ++j)
      for (PetscInt i = info.start[0]; i < info.start[0] + info.size[0]; ++i, ++cell)
      {
        MatStencil row{};
        row.i = i, row.j = j, row.k = k;
        cols.clear();
        values.clear();
        const auto add = [&](const std::array<PetscInt, 3> & offset, PetscScalar value)
        {
          std::array<PetscInt, 3> c{{i + offset[0], j + offset[1], k + offset[2]}};
          Real factor = 1.0;
          for (const auto a : make_range(dim))
            factor *= foldIndex(c[a], info.M[a], periodic[a], faces, marlinAxis(dim, a), h);
          MatStencil col{};
          col.i = c[0], col.j = c[1], col.k = c[2];
          cols.push_back(col);
          values.push_back(factor * value);
        };
        for (const auto & [offset, value] : merged)
          add(offset, value);
        for (const auto & entry : field_entries)
          add(entry.offset, entry.value * entry.field.data_ptr<double>()[cell]);
        row_fn(row, cols, values);
      }
}
}

Real
FaceGhosts::factor(unsigned int face, unsigned int layer, Real h) const
{
  // the ghost in layer l and its mirror are (2l + 1) h apart
  return TensorRobinBC::ghostCoefficientOnU0(ab[face][0], ab[face][1], (2.0 * layer + 1.0) * h);
}

Partition
gatherPartition(const DomainAction & domain)
{
  Partition partition;
  partition.dim = domain.getDim();
  partition.n = domain.getGridSize();
  partition.parts = domain.getRealSpacePartitions();
  partition.index = domain.getRealSpaceIndex();
  partition.periodic = domain.getPeriodicDirections();
  for (const auto d : make_range(3))
    partition.h[d] = domain.getGridSpacing()(d);
  partition.comm = domain.getMPIComm();

  int size;
  MPI_Comm_size(partition.comm, &size);

  // every rank contributes (index, owned size) per axis, which gives the slab sizes
  const auto & local = domain.getLocalGridSize();
  std::array<int64_t, 6> mine;
  for (const auto d : make_range(3))
  {
    mine[d] = partition.index[d];
    mine[3 + d] = local[d];
  }
  std::vector<int64_t> all(6 * size);
  MPI_Allgather(mine.data(), 6, MPI_INT64_T, all.data(), 6, MPI_INT64_T, partition.comm);
  for (const auto d : make_range(3))
  {
    partition.counts[d].assign(partition.parts[d], 0);
    for (const auto r : make_range(size))
      partition.counts[d][all[6 * r + d]] = all[6 * r + 3 + d];
  }
  return partition;
}

MPI_Comm
dmdaComm(const Partition & partition)
{
  // DMDA rank = i_0 + p_0 (i_1 + p_1 i_2) in DMDA axes, DMDA axis a = Marlin axis dim - 1 - a
  int key = 0;
  for (int a = int(partition.dim) - 1; a >= 0; --a)
  {
    const auto m = marlinAxis(partition.dim, a);
    key = key * int(partition.parts[m]) + int(partition.index[m]);
  }
  MPI_Comm comm;
  MPI_Comm_split(partition.comm, 0, key, &comm);
  return comm;
}

PetscInt
stencilWidth(const std::vector<FDStencilEntry> & stencil)
{
  PetscInt width = 1;
  for (const auto & entry : stencil)
    for (const auto o : entry.offset)
      width = std::max(width, PetscInt(std::abs(o)));
  return width;
}

bool
needsBoxStencil(const std::vector<FDStencilEntry> & stencil)
{
  for (const auto & entry : stencil)
    if (std::count_if(entry.offset.begin(), entry.offset.end(), [](int o) { return o != 0; }) > 1)
      return true;
  return false;
}

unsigned int
countLevels(const Partition & partition, PetscInt stencil_width)
{
  unsigned int levels = 1;
  while (true)
  {
    for (const auto d : make_range(partition.dim))
    {
      const auto extent = partition.n[d] >> (levels - 1);
      if (extent % 2 || extent / 2 < coarsest_extent)
        return levels;
      // DMCoarsen halves every slab (rounding as needed); each rank must keep a full stencil
      for (const auto count : partition.counts[d])
        if ((count >> levels) < stencil_width)
          return levels;
    }
    ++levels;
  }
}

Hierarchy::~Hierarchy()
{
  for (auto & P : interpolations)
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&P));
  for (auto & da : dms)
    PetscCallAbort(PETSC_COMM_WORLD, DMDestroy(&da));
}

void
buildHierarchy(const Partition & partition,
               MPI_Comm comm,
               unsigned int levels,
               PetscInt stencil_width,
               bool box,
               const FaceGhosts & faces,
               Hierarchy & hierarchy)
{
  const auto dim = partition.dim;
  const auto stencil_type = box ? DMDA_STENCIL_BOX : DMDA_STENCIL_STAR;

  // global extents, ranks, slab sizes and boundary types in DMDA axis order
  std::array<PetscInt, 3> M{{1, 1, 1}};
  std::array<PetscMPIInt, 3> procs{{1, 1, 1}};
  std::array<std::vector<PetscInt>, 3> counts;
  std::array<DMBoundaryType, 3> bt{
      {DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC}};
  for (const auto a : make_range(dim))
  {
    const auto m = marlinAxis(dim, a);
    M[a] = PetscInt(partition.n[m]);
    procs[a] = PetscMPIInt(partition.parts[m]);
    counts[a] = partition.counts[m];
    bt[a] = partition.periodic[m] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE;
  }

  DM fine;
  if (dim == 1)
    PetscCallAbort(comm,
                   DMDACreate1d(comm, bt[0], M[0], 1, stencil_width, counts[0].data(), &fine));
  else if (dim == 2)
    PetscCallAbort(comm,
                   DMDACreate2d(comm,
                                bt[0],
                                bt[1],
                                stencil_type,
                                M[0],
                                M[1],
                                procs[0],
                                procs[1],
                                1,
                                stencil_width,
                                counts[0].data(),
                                counts[1].data(),
                                &fine));
  else
    PetscCallAbort(comm,
                   DMDACreate3d(comm,
                                bt[0],
                                bt[1],
                                bt[2],
                                stencil_type,
                                M[0],
                                M[1],
                                M[2],
                                procs[0],
                                procs[1],
                                procs[2],
                                1,
                                stencil_width,
                                counts[0].data(),
                                counts[1].data(),
                                counts[2].data(),
                                &fine));
  // DMCoarsen halves a non-periodic axis cell-centred (n -> n/2, as a periodic one) only
  // for Q0 interpolation; otherwise vertex-centred (n -> 1 + (n-1)/2). The interpolation
  // itself is cellCentredInterpolation, not the DMDA's.
  PetscCallAbort(comm, DMDASetInterpolationType(fine, DMDA_Q0));
  PetscCallAbort(comm, DMSetUp(fine));
  hierarchy.dms.push_back(fine);

  // the fine DMDA must own exactly this rank's Marlin block
  const auto info = getInfo(fine);
  bool owned_match = true;
  for (const auto a : make_range(dim))
  {
    const auto m = marlinAxis(dim, a);
    PetscInt offset = 0;
    for (const auto i : make_range(partition.index[m]))
      offset += partition.counts[m][i];
    owned_match = owned_match && info.start[a] == offset &&
                  info.size[a] == partition.counts[m][partition.index[m]];
  }
  PetscCheckAbort(
      owned_match, comm, PETSC_ERR_PLIB, "The DMDA ownership differs from Marlin's partition");

  // coarse levels and the interpolations between neighbouring levels
  std::vector<DM> coarse(levels - 1);
  if (levels > 1)
    PetscCallAbort(comm, DMCoarsenHierarchy(fine, levels - 1, coarse.data()));
  for (const auto l : index_range(coarse))
  {
    hierarchy.dms.push_back(coarse[l]);
    // level l + 1 has spacing 2^(l + 1) h
    std::vector<Real> h_coarse(dim);
    for (const auto d : make_range(dim))
      h_coarse[d] = partition.h[d] * Real(2 << l);
    hierarchy.interpolations.push_back(
        cellCentredInterpolation(coarse[l], hierarchy.dms[l], faces, h_coarse));
  }
}

void
assembleStencilOperator(DM da,
                        const std::vector<FDStencilEntry> & stencil,
                        PetscReal scale,
                        const FaceGhosts & faces,
                        const std::vector<Real> & h,
                        Mat B)
{
  const MPI_Comm comm = PetscObjectComm((PetscObject)da);

  // a MATPREALLOCATOR (createStencilMatrix) is never assembled and has no values
  PetscBool assembled;
  PetscCallAbort(comm, MatAssembled(B, &assembled));
  if (assembled)
    PetscCallAbort(comm, MatZeroEntries(B));
  // periodic wrap of indices outside the domain is done by the DMDA; columns reached
  // twice (offsets that wrap onto each other on a small periodic grid, or folded onto
  // the same cell at a boundary face) are added, hence ADD_VALUES
  forEachFoldedRow(
      da,
      stencil,
      scale,
      faces,
      h,
      [&](const MatStencil & row,
          const std::vector<MatStencil> & cols,
          const std::vector<PetscScalar> & values)
      {
        PetscCallAbort(
            comm,
            MatSetValuesStencil(
                B, 1, &row, PetscInt(cols.size()), cols.data(), values.data(), ADD_VALUES));
      });
  PetscCallAbort(comm, MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(comm, MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
}

void
addIdentityOnZeroRows(Mat B)
{
  const MPI_Comm comm = PetscObjectComm((PetscObject)B);
  Vec diag;
  PetscCallAbort(comm, MatCreateVecs(B, nullptr, &diag));
  PetscCallAbort(comm, MatGetDiagonal(B, diag));
  PetscInt local, zero_rows = 0;
  PetscCallAbort(comm, VecGetLocalSize(diag, &local));
  PetscScalar * array;
  PetscCallAbort(comm, VecGetArray(diag, &array));
  PetscReal smallest = std::numeric_limits<PetscReal>::max();
  for (const auto i : make_range(local))
    if (array[i] != 0.0)
      smallest = std::min(smallest, PetscReal(array[i]));
  PetscCallMPIAbort(comm, MPI_Allreduce(MPI_IN_PLACE, &smallest, 1, MPIU_REAL, MPI_MIN, comm));
  for (const auto i : make_range(local))
    if (array[i] == 0.0)
    {
      array[i] = smallest;
      ++zero_rows;
    }
    else
      array[i] = 0.0;
  PetscCallAbort(comm, VecRestoreArray(diag, &array));
  PetscCallMPIAbort(comm, MPI_Allreduce(MPI_IN_PLACE, &zero_rows, 1, MPIU_INT, MPI_SUM, comm));
  if (zero_rows > 0)
    PetscCallAbort(comm, MatDiagonalSet(B, diag, ADD_VALUES));
  PetscCallAbort(comm, VecDestroy(&diag));
}

Real
jacobiSpectrumBound(DM da,
                    const std::vector<FDStencilEntry> & stencil,
                    const FaceGhosts & faces,
                    const std::vector<Real> & h)
{
  Real bound = 0.0;
  forEachFoldedRow(da,
                   stencil,
                   1.0,
                   faces,
                   h,
                   [&](const MatStencil & row,
                       const std::vector<MatStencil> & cols,
                       const std::vector<PetscScalar> & values)
                   {
                     // the matrix entries: values on the same column (folded or wrapped)
                     // are added, as in assembleStencilOperator
                     std::map<std::array<PetscInt, 3>, Real> row_values;
                     for (const auto q : index_range(cols))
                       row_values[{{cols[q].i, cols[q].j, cols[q].k}}] += values[q];
                     Real centre = 0.0, off_diagonal = 0.0;
                     for (const auto & [col, value] : row_values)
                       if (col == std::array<PetscInt, 3>{{row.i, row.j, row.k}})
                         centre += value;
                       else
                         off_diagonal += std::abs(value);
                     // an inactive cell's zero row becomes an identity row (bound 1)
                     bound = centre == 0.0 && off_diagonal == 0.0
                                 ? std::max(bound, 1.0)
                                 : std::max(bound, 1.0 + off_diagonal / centre);
                   });
  MPI_Allreduce(MPI_IN_PLACE, &bound, 1, MPI_DOUBLE, MPI_MAX, PetscObjectComm((PetscObject)da));
  return bound;
}

torch::Tensor
foldedDiagonal(const std::vector<FDStencilEntry> & stencil,
               unsigned int dim,
               const std::array<int64_t, 3> & n,
               const std::array<bool, 3> & periodic,
               const FaceGhosts & faces,
               const std::vector<Real> & h,
               const std::array<int64_t, 3> & begin,
               const std::array<int64_t, 3> & size,
               const torch::TensorOptions & options)
{
  std::vector<int64_t> shape(size.begin(), size.begin() + dim);
  auto diag = torch::zeros(shape, options);
  for (const auto & entry : stencil)
  {
    // the entry's weight on the diagonal is a product over axes of its weight along each
    auto weight = entry.field.defined() ? entry.coefficient * entry.field.reshape(shape).to(options)
                                        : torch::full(shape, entry.coefficient, options);
    for (const auto d : make_range(dim))
    {
      const auto o = int64_t(entry.offset[d]);
      std::vector<Real> along(size[d], 0.0);
      for (const auto q : make_range(size[d]))
      {
        const auto i = begin[d] + q;
        const auto c = i + o;
        if (periodic[d])
          along[q] = o % n[d] == 0 ? 1.0 : 0.0;
        else if (o == 0)
          along[q] = 1.0;
        else if (c < 0 && -c - 1 == i)
          along[q] = faces.factor(2 * d, unsigned(i), h[d]);
        else if (c >= n[d] && n[d] - 1 - (c - n[d]) == i)
          along[q] = faces.factor(2 * d + 1, unsigned(c - n[d]), h[d]);
      }
      std::vector<int64_t> axis_shape(dim, 1);
      axis_shape[d] = size[d];
      weight = weight * torch::tensor(along, torch::TensorOptions().dtype(torch::kFloat64))
                            .reshape(axis_shape)
                            .to(options);
    }
    diag += weight;
  }
  return diag;
}

Mat
createStencilMatrix(DM da, const std::vector<FDStencilEntry> & stencil)
{
  const MPI_Comm comm = PetscObjectComm((PetscObject)da);

  // the DMDA preallocates the whole (2w + 1)^dim box of a box stencil (729 entries per
  // row for the 5-point biharmonic in 3D, of which 73 are used), so it only provides
  // the layout and the stencil numbering here
  Mat B;
  PetscCallAbort(comm, DMSetMatrixPreallocateSkip(da, PETSC_TRUE));
  PetscCallAbort(comm, DMCreateMatrix(da, &B));
  PetscCallAbort(comm, DMSetMatrixPreallocateSkip(da, PETSC_FALSE));

  // record the stencil's columns in a preallocator with the same numbering
  Mat preallocator;
  PetscInt m, n, M, N;
  PetscCallAbort(comm, MatGetLocalSize(B, &m, &n));
  PetscCallAbort(comm, MatGetSize(B, &M, &N));
  PetscCallAbort(comm, MatCreate(comm, &preallocator));
  PetscCallAbort(comm, MatSetType(preallocator, MATPREALLOCATOR));
  PetscCallAbort(comm, MatSetSizes(preallocator, m, n, M, N));
  PetscCallAbort(comm, MatSetUp(preallocator));
  ISLocalToGlobalMapping ltog;
  PetscCallAbort(comm, DMGetLocalToGlobalMapping(da, &ltog));
  PetscCallAbort(comm, MatSetLocalToGlobalMapping(preallocator, ltog, ltog));
  const auto info = getInfo(da);
  std::array<PetscInt, 3> starts, dims;
  PetscCallAbort(
      comm,
      DMDAGetGhostCorners(da, &starts[0], &starts[1], &starts[2], &dims[0], &dims[1], &dims[2]));
  PetscCallAbort(comm, MatSetStencil(preallocator, info.dim, dims.data(), starts.data(), 1));
  // the folded columns are the same for any ghost factors (unit factors here, so no
  // entry cancels) and any per-cell fields (dropped, they need not be on this level)
  FaceGhosts unit_faces;
  for (auto & ab : unit_faces.ab)
    ab = {0.0, 1.0};
  auto columns = stencil;
  for (auto & entry : columns)
    entry.field = torch::Tensor();
  assembleStencilOperator(
      da, columns, 1.0, unit_faces, std::vector<Real>(getInfo(da).dim, 1.0), preallocator);

  // preallocate exactly those columns and insert zeros there; any other column is an error
  PetscCallAbort(comm, MatPreallocatorPreallocate(preallocator, PETSC_TRUE, B));
  PetscCallAbort(comm, MatDestroy(&preallocator));
  return B;
}

void
restrictCellMean(DM coarse, DM fine, Vec fine_values, Vec coarse_values)
{
  const MPI_Comm comm = PetscObjectComm((PetscObject)fine);
  // the hierarchy's DMDAs interpolate with DMDA_Q0 (buildHierarchy): each fine cell takes
  // its coarse parent's value, so the transpose sums the children
  Mat injection;
  PetscCallAbort(comm, DMCreateInterpolation(coarse, fine, &injection, nullptr));
  PetscCallAbort(comm, MatMultTranspose(injection, fine_values, coarse_values));
  PetscCallAbort(comm, VecScale(coarse_values, 1.0 / Real(PetscInt(1) << getInfo(fine).dim)));
  PetscCallAbort(comm, MatDestroy(&injection));
}

torch::Tensor
ownedWithHalo(DM da, Vec v)
{
  const MPI_Comm comm = PetscObjectComm((PetscObject)da);
  const auto info = getInfo(da);
  const auto dim = unsigned(info.dim);
  std::array<PetscInt, 3> gs, gm;
  PetscCallAbort(comm, DMDAGetGhostCorners(da, &gs[0], &gs[1], &gs[2], &gm[0], &gm[1], &gm[2]));

  Vec local;
  PetscCallAbort(comm, DMGetLocalVector(da, &local));
  PetscCallAbort(comm, VecSet(local, 0.0));
  PetscCallAbort(comm, DMGlobalToLocal(da, v, INSERT_VALUES, local));
  // the local vector numbers x fastest, so in Marlin axis order it is row-major
  std::vector<int64_t> shape(dim);
  for (const auto m : make_range(dim))
    shape[m] = gm[marlinAxis(dim, m)];
  const PetscScalar * a;
  PetscCallAbort(comm, VecGetArrayRead(local, &a));
  auto t = torch::from_blob(
               const_cast<PetscScalar *>(a), shape, torch::TensorOptions().dtype(torch::kFloat64))
               .clone();
  PetscCallAbort(comm, VecRestoreArrayRead(local, &a));
  PetscCallAbort(comm, DMRestoreLocalVector(da, &local));

  for (const auto m : make_range(dim))
  {
    const auto ax = marlinAxis(dim, m);
    // without a ghost cell on a side (a domain face) the nearest owned cell is repeated
    const bool lower = gs[ax] < info.start[ax];
    const bool upper = gs[ax] + gm[ax] > info.start[ax] + info.size[ax];
    t = t.narrow(m, info.start[ax] - gs[ax] - (lower ? 1 : 0), info.size[ax] + lower + upper);
    if (!lower)
      t = torch::cat({t.narrow(m, 0, 1), t}, m);
    if (!upper)
      t = torch::cat({t, t.narrow(m, t.size(m) - 1, 1)}, m);
  }
  return t.contiguous();
}

Mat
cellCentredInterpolation(DM coarse,
                         DM fine,
                         const FaceGhosts & faces,
                         const std::vector<Real> & h_coarse)
{
  const MPI_Comm comm = PetscObjectComm((PetscObject)fine);
  const auto fine_info = getInfo(fine);
  const auto coarse_info = getInfo(coarse);
  const auto dim = unsigned(fine_info.dim);
  const auto periodic = periodicAxes(fine);

  // slab boundaries of the coarse ownership ranges, per DMDA axis
  std::array<const PetscInt *, 3> ranges{{nullptr, nullptr, nullptr}};
  PetscCallAbort(comm, DMDAGetOwnershipRanges(coarse, &ranges[0], &ranges[1], &ranges[2]));
  std::array<std::vector<PetscInt>, 3> bounds;
  std::array<std::vector<PetscInt>, 3> slab_sizes;
  for (const auto a : make_range(3u))
  {
    const auto procs = coarse_info.procs[a];
    slab_sizes[a].assign(procs, 1);
    if (a < dim)
      slab_sizes[a].assign(ranges[a], ranges[a] + procs);
    bounds[a].assign(procs + 1, 0);
    for (const auto p : make_range(procs))
      bounds[a][p + 1] = bounds[a][p] + slab_sizes[a][p];
  }

  // DMDA global numbering: rank blocks in rank order p_0 + P_0 (p_1 + P_1 p_2),
  // x fastest inside a block; offsets[q] is the first global index of rank q
  const auto & procs = coarse_info.procs;
  std::vector<PetscInt> offsets(procs[0] * procs[1] * procs[2] + 1, 0);
  for (const auto q : make_range(PetscInt(offsets.size()) - 1))
    offsets[q + 1] = offsets[q] + slab_sizes[0][q % procs[0]] *
                                      slab_sizes[1][(q / procs[0]) % procs[1]] *
                                      slab_sizes[2][q / (procs[0] * procs[1])];
  const auto global_column = [&](std::array<PetscInt, 3> c)
  {
    std::array<PetscInt, 3> p{{0, 0, 0}};
    for (const auto a : make_range(dim))
    {
      c[a] = (c[a] + coarse_info.M[a]) % coarse_info.M[a];
      p[a] = std::upper_bound(bounds[a].begin(), bounds[a].end(), c[a]) - bounds[a].begin() - 1;
    }
    const PetscInt rank = p[0] + procs[0] * (p[1] + procs[1] * p[2]);
    const auto & s = slab_sizes;
    return offsets[rank] +
           ((c[2] - bounds[2][p[2]]) * s[1][p[1]] + (c[1] - bounds[1][p[1]])) * s[0][p[0]] +
           (c[0] - bounds[0][p[0]]);
  };

  const PetscInt fine_local = fine_info.size[0] * fine_info.size[1] * fine_info.size[2];
  const PetscInt coarse_local = coarse_info.size[0] * coarse_info.size[1] * coarse_info.size[2];
  // 2^dim coarse cells contribute to each fine cell
  const PetscInt entries = PetscInt(1) << dim;
  Mat P;
  PetscCallAbort(comm,
                 MatCreateAIJ(comm,
                              fine_local,
                              coarse_local,
                              PETSC_DETERMINE,
                              PETSC_DETERMINE,
                              entries,
                              nullptr,
                              entries,
                              nullptr,
                              &P));
  PetscInt row = 0;
  PetscCallAbort(comm, MatGetOwnershipRange(P, &row, nullptr));

  std::vector<PetscInt> cols(entries);
  std::vector<PetscScalar> vals(entries);
  const auto & start = fine_info.start;
  const auto & size = fine_info.size;
  for (PetscInt k = start[2]; k < start[2] + size[2]; ++k)
    for (PetscInt j = start[1]; j < start[1] + size[1]; ++j)
      for (PetscInt i = start[0]; i < start[0] + size[0]; ++i, ++row)
      {
        const std::array<PetscInt, 3> f{{i, j, k}};
        // tensor product of the per-axis parent and neighbour weights; a neighbour
        // outside a boundary face is the coarse ghost, factor * parent, so its weight
        // goes to the parent
        for (const auto e : make_range(entries))
        {
          std::array<PetscInt, 3> c{{0, 0, 0}};
          PetscScalar w = 1.0;
          for (const auto a : make_range(dim))
          {
            const PetscInt parent = f[a] / 2;
            const bool neighbour = (e >> a) & 1;
            c[a] = neighbour ? (f[a] % 2 ? parent + 1 : parent - 1) : parent;
            w *= neighbour ? 0.25 : 0.75;
            w *=
                foldIndex(c[a], coarse_info.M[a], periodic[a], faces, marlinAxis(dim, a), h_coarse);
          }
          cols[e] = global_column(c);
          vals[e] = w;
        }
        // folded neighbours share the parent's column
        PetscCallAbort(comm,
                       MatSetValues(P, 1, &row, entries, cols.data(), vals.data(), ADD_VALUES));
      }
  PetscCallAbort(comm, MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(comm, MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  return P;
}

} // namespace PetscFD
