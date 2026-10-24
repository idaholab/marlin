/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/
#include "PetscFDHierarchy.h"
#include "DomainAction.h"

#include "libmesh/int_range.h"

#include <algorithm>
#include <cstdlib>
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
}

Partition
gatherPartition(const DomainAction & domain)
{
  Partition partition;
  partition.dim = domain.getDim();
  partition.n = domain.getGridSize();
  partition.parts = domain.getRealSpacePartitions();
  partition.index = domain.getRealSpaceIndex();
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
               Hierarchy & hierarchy)
{
  const auto dim = partition.dim;
  const auto stencil_type = box ? DMDA_STENCIL_BOX : DMDA_STENCIL_STAR;
  const auto periodic = DM_BOUNDARY_PERIODIC;

  // global extents, ranks and slab sizes in DMDA axis order
  std::array<PetscInt, 3> M{{1, 1, 1}};
  std::array<PetscMPIInt, 3> procs{{1, 1, 1}};
  std::array<std::vector<PetscInt>, 3> counts;
  for (const auto a : make_range(dim))
  {
    const auto m = marlinAxis(dim, a);
    M[a] = PetscInt(partition.n[m]);
    procs[a] = PetscMPIInt(partition.parts[m]);
    counts[a] = partition.counts[m];
  }

  DM fine;
  if (dim == 1)
    PetscCallAbort(comm,
                   DMDACreate1d(comm, periodic, M[0], 1, stencil_width, counts[0].data(), &fine));
  else if (dim == 2)
    PetscCallAbort(comm,
                   DMDACreate2d(comm,
                                periodic,
                                periodic,
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
                                periodic,
                                periodic,
                                periodic,
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
    hierarchy.interpolations.push_back(cellCentredInterpolation(coarse[l], hierarchy.dms[l]));
  }
}

void
assembleStencilOperator(DM da, const std::vector<FDStencilEntry> & stencil, PetscReal scale, Mat B)
{
  const MPI_Comm comm = PetscObjectComm((PetscObject)da);
  const auto info = getInfo(da);
  const auto dim = unsigned(info.dim);

  // merge entries with the same offset, converted to DMDA axis order
  std::map<std::array<PetscInt, 3>, PetscScalar> merged;
  for (const auto & entry : stencil)
  {
    std::array<PetscInt, 3> offset{{0, 0, 0}};
    for (const auto a : make_range(dim))
      offset[a] = entry.offset[marlinAxis(dim, a)];
    merged[offset] += scale * entry.coefficient;
  }
  std::vector<std::array<PetscInt, 3>> offsets;
  std::vector<PetscScalar> values;
  for (const auto & [offset, value] : merged)
  {
    offsets.push_back(offset);
    values.push_back(value);
  }

  // a MATPREALLOCATOR (createStencilMatrix) is never assembled and has no values
  PetscBool assembled;
  PetscCallAbort(comm, MatAssembled(B, &assembled));
  if (assembled)
    PetscCallAbort(comm, MatZeroEntries(B));
  std::vector<MatStencil> cols(offsets.size());
  for (PetscInt k = info.start[2]; k < info.start[2] + info.size[2]; ++k)
    for (PetscInt j = info.start[1]; j < info.start[1] + info.size[1]; ++j)
      for (PetscInt i = info.start[0]; i < info.start[0] + info.size[0]; ++i)
      {
        MatStencil row{};
        row.i = i, row.j = j, row.k = k;
        // periodic wrap of indices outside the domain is done by the DMDA; offsets that
        // wrap onto the same column (e.g. +-4 on 8 cells) are added, hence ADD_VALUES
        for (const auto q : index_range(offsets))
        {
          cols[q] = row;
          cols[q].i += offsets[q][0];
          cols[q].j += offsets[q][1];
          cols[q].k += offsets[q][2];
        }
        PetscCallAbort(
            comm,
            MatSetValuesStencil(
                B, 1, &row, PetscInt(cols.size()), cols.data(), values.data(), ADD_VALUES));
      }
  PetscCallAbort(comm, MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(comm, MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
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
  assembleStencilOperator(da, stencil, 1.0, preallocator);

  // preallocate exactly those columns and insert zeros there; any other column is an error
  PetscCallAbort(comm, MatPreallocatorPreallocate(preallocator, PETSC_TRUE, B));
  PetscCallAbort(comm, MatDestroy(&preallocator));
  return B;
}

Mat
cellCentredInterpolation(DM coarse, DM fine)
{
  const MPI_Comm comm = PetscObjectComm((PetscObject)fine);
  const auto fine_info = getInfo(fine);
  const auto coarse_info = getInfo(coarse);
  const auto dim = unsigned(fine_info.dim);

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
        // tensor product of the per-axis parent and neighbour weights
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
          }
          cols[e] = global_column(c);
          vals[e] = w;
        }
        PetscCallAbort(comm,
                       MatSetValues(P, 1, &row, entries, cols.data(), vals.data(), INSERT_VALUES));
      }
  PetscCallAbort(comm, MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(comm, MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  return P;
}

} // namespace PetscFD
