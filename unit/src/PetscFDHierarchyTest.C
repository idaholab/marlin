/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/
#include "PetscFDHierarchy.h"
#include "FiniteDifferenceLaplacian.h"
#include "FiniteDifferenceBiharmonic.h"
#include "RobinGhost.h"
#include "gtest/gtest.h"

#include "libmesh/int_range.h"

#include <cmath>
#include <array>
#include <functional>
#include <vector>

namespace
{
/**
 * A partition of n cells per axis over the ranks of MPI_COMM_WORLD: 4 ranks split
 * Marlin axes 0 and 1 in two, any other count splits axis 0. Slab sizes follow
 * DomainAction::partitionHepler (32 cells on 3 ranks: 10, 11, 11). The tests are
 * meant for at most 4 ranks: on more, the small grids leave a rank fewer coarse cells
 * than the stencil width and the expected level counts no longer hold.
 */
PetscFD::Partition
uniformPartition(unsigned int dim, int64_t n)
{
  PetscFD::Partition p;
  p.dim = dim;
  p.comm = MPI_COMM_WORLD;
  int size, rank;
  MPI_Comm_size(p.comm, &size);
  MPI_Comm_rank(p.comm, &rank);
  p.parts = {1, 1, 1};
  if (size == 4 && dim > 1)
    p.parts = {2, 2, 1};
  else
    p.parts[0] = size;
  // Marlin's rank order: ix + px (iy + py iz)
  p.index = {unsigned(rank) % p.parts[0], (unsigned(rank) / p.parts[0]) % p.parts[1], 0};
  for (const auto d : make_range(3))
  {
    p.n[d] = d < dim ? n : 1;
    p.counts[d].clear();
    int64_t assigned = 0;
    for (const auto i : make_range(p.parts[d]))
    {
      p.counts[d].push_back(PetscInt((p.n[d] - assigned) / (p.parts[d] - i)));
      assigned += p.counts[d].back();
    }
  }
  return p;
}

/// the same partition with boundary faces (spacing 1/n) on the Marlin axes not in periodic
PetscFD::Partition
boundedPartition(unsigned int dim, int64_t n, std::array<bool, 3> periodic)
{
  auto p = uniformPartition(dim, n);
  p.periodic = periodic;
  p.h = {1.0 / n, 1.0 / n, 1.0 / n};
  return p;
}

/// every face Dirichlet (a = 1, b = 0) or every face Neumann (a = 0, b = 1)
PetscFD::FaceGhosts
uniformFaces(Real a, Real b)
{
  PetscFD::FaceGhosts faces;
  for (auto & ab : faces.ab)
    ab = {a, b};
  return faces;
}

/// fill v (on da) with f(x) at the cell centres, x in Marlin axis order on [0, 1)^dim
void
fillVec(DM da, Vec v, const std::function<double(const std::array<double, 3> &)> & f)
{
  PetscInt dim, M[3] = {1, 1, 1}, s[3], m[3];
  PetscCallAbort(PETSC_COMM_WORLD,
                 DMDAGetInfo(da, &dim, &M[0], &M[1], &M[2], 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCallAbort(PETSC_COMM_WORLD, DMDAGetCorners(da, &s[0], &s[1], &s[2], &m[0], &m[1], &m[2]));
  PetscScalar * a;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(v, &a));
  PetscInt q = 0;
  for (PetscInt k = s[2]; k < s[2] + m[2]; ++k)
    for (PetscInt j = s[1]; j < s[1] + m[1]; ++j)
      for (PetscInt i = s[0]; i < s[0] + m[0]; ++i)
      {
        const PetscInt idx[3] = {i, j, k};
        std::array<double, 3> x{{0, 0, 0}};
        // DMDA axis a is Marlin axis dim - 1 - a
        for (PetscInt ax = 0; ax < dim; ++ax)
          x[dim - 1 - ax] = (idx[ax] + 0.5) / M[ax];
        a[q++] = f(x);
      }
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(v, &a));
}

double
maxDiff(Vec a, Vec b)
{
  Vec d;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(a, &d));
  PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(d, -1.0, b, a));
  PetscReal norm;
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(d, NORM_INFINITY, &norm));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&d));
  return norm;
}
}

// Level count: halve while every extent is even and stays >= 8 and every rank keeps a
// full stencil width.
TEST(PetscFDHierarchyTest, LevelCount)
{
  // 64 -> 32 -> 16 -> 8
  EXPECT_EQ(PetscFD::countLevels(uniformPartition(2, 64), 1), 4u);
  // 96 -> 48 -> 24 -> 12 (6 < 8)
  EXPECT_EQ(PetscFD::countLevels(uniformPartition(2, 96), 1), 4u);
  // odd extents cannot be coarsened
  EXPECT_EQ(PetscFD::countLevels(uniformPartition(3, 18), 1), 2u);
  EXPECT_EQ(PetscFD::countLevels(uniformPartition(1, 64), 2), 4u);
}

// The cell-centred interpolation: every row sums to 1, constants are reproduced, and it
// is exact for linear fields (away from the periodic seam, where a linear field jumps).
TEST(PetscFDHierarchyTest, CellCentredInterpolation)
{
  for (const unsigned int dim : {1u, 2u, 3u})
  {
    const int64_t n = dim == 3 ? 16 : 32;
    const auto partition = uniformPartition(dim, n);
    MPI_Comm comm = PetscFD::dmdaComm(partition);
    {
      PetscFD::Hierarchy h;
      PetscFD::buildHierarchy(partition, comm, 2, 1, false, {}, h);
      ASSERT_EQ(h.dms.size(), 2u);
      ASSERT_EQ(h.interpolations.size(), 1u);
      Mat P = h.interpolations[0];

      Vec coarse, fine, expected;
      PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[1], &coarse));
      PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[0], &fine));
      PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[0], &expected));

      // constant
      PetscCallAbort(PETSC_COMM_WORLD, VecSet(coarse, 1.0));
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(P, coarse, fine));
      PetscCallAbort(PETSC_COMM_WORLD, VecSet(expected, 1.0));
      EXPECT_LT(maxDiff(fine, expected), 1e-14) << "dim " << dim;

      // linear in every axis, compared at fine cells whose stencil does not cross the seam
      const auto linear = [&](const std::array<double, 3> & x)
      { return 1.0 + 2.0 * x[0] - 3.0 * x[1] + 0.5 * x[2]; };
      fillVec(h.dms[1], coarse, linear);
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(P, coarse, fine));
      fillVec(h.dms[0], expected, linear);
      const auto interior = [&](const std::array<double, 3> & x)
      {
        // fine cells in the outermost quarter coarse cell of each end use a wrapped neighbour
        bool inside = true;
        for (const auto d : make_range(dim))
          inside = inside && x[d] > 1.0 / n && x[d] < 1.0 - 1.0 / n;
        return inside ? 1.0 : 0.0;
      };
      Vec mask;
      PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[0], &mask));
      fillVec(h.dms[0], mask, interior);
      PetscCallAbort(PETSC_COMM_WORLD, VecPointwiseMult(fine, fine, mask));
      PetscCallAbort(PETSC_COMM_WORLD, VecPointwiseMult(expected, expected, mask));
      EXPECT_LT(maxDiff(fine, expected), 1e-13) << "dim " << dim;

      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&mask));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&coarse));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&fine));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&expected));
    }
    MPI_Comm_free(&comm);
  }
}

// The fine DMDA owns this rank's Marlin block, and its global vector order is the
// row-major order of that block (Marlin's last axis fastest), so a contiguous owned
// tensor can be wrapped as the DMDA vector without reordering.
TEST(PetscFDHierarchyTest, FineDMDAMatchesMarlinBlock)
{
  for (const unsigned int dim : {1u, 2u, 3u})
  {
    const int64_t n = dim == 3 ? 8 : 16;
    const auto partition = uniformPartition(dim, n);
    MPI_Comm comm = PetscFD::dmdaComm(partition);
    {
      // buildHierarchy aborts if the DMDA ownership differs from the partition
      PetscFD::Hierarchy h;
      PetscFD::buildHierarchy(partition, comm, 1, 1, false, {}, h);

      // global cell index i_0 n^2 + i_1 n + i_2 (Marlin order), via the DMDA coordinates
      Vec v;
      PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[0], &v));
      fillVec(h.dms[0],
              v,
              [&](const std::array<double, 3> & x)
              {
                double id = 0.0;
                for (const auto d : make_range(dim))
                  id = id * n + std::floor(x[d] * n);
                return id;
              });

      // the same indices from Marlin's partition, in row-major order of the owned block
      std::array<int64_t, 3> start{{0, 0, 0}}, size{{1, 1, 1}};
      for (const auto d : make_range(dim))
      {
        for (const auto i : make_range(partition.index[d]))
          start[d] += partition.counts[d][i];
        size[d] = partition.counts[d][partition.index[d]];
      }
      const PetscScalar * a;
      PetscInt local;
      PetscCallAbort(PETSC_COMM_WORLD, VecGetLocalSize(v, &local));
      ASSERT_EQ(local, size[0] * size[1] * size[2]);
      PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(v, &a));
      PetscInt q = 0;
      for (const auto i0 : make_range(size[0]))
        for (const auto i1 : make_range(size[1]))
          for (const auto i2 : make_range(size[2]))
          {
            const std::array<int64_t, 3> g{{start[0] + i0, start[1] + i1, start[2] + i2}};
            double id = 0.0;
            for (const auto d : make_range(dim))
              id = id * n + g[d];
            EXPECT_EQ(a[q++], id) << "dim " << dim;
          }
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(v, &a));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&v));
    }
    MPI_Comm_free(&comm);
  }
}

// The assembled coarse operator is the stencil rediscretised at the coarse spacing:
// a sum of single-axis periodic modes is mapped to the sum of the modes, each scaled
// by the discrete symbol of its axis at that spacing. Unequal domain lengths and mode
// numbers per axis catch any axis mix-up.
TEST(PetscFDHierarchyTest, AssembledOperatorIsRediscretisation)
{
  const double pi = M_PI;
  // domain lengths and mode numbers per Marlin axis
  const std::array<double, 3> L{{1.0, 2.0, 0.5}};
  const std::array<int, 3> k{{1, 2, 3}};
  for (const unsigned int dim : {1u, 2u, 3u})
    for (const bool wide : {false, true})
    {
      const int64_t n = dim == 3 ? 16 : 32;
      const auto partition = uniformPartition(dim, n);
      MPI_Comm comm = PetscFD::dmdaComm(partition);
      {
        PetscFD::Hierarchy h;
        PetscFD::buildHierarchy(partition, comm, 2, wide ? 2 : 1, false, {}, h);

        // coarse level: n / 2 cells per axis
        const int64_t nc = n / 2;
        std::vector<Real> dx(dim);
        for (const auto d : make_range(dim))
          dx[d] = L[d] / nc;
        const auto stencil = FiniteDifferenceLaplacian::buildStencil(dim, wide, dx, -1.0);

        Mat B = PetscFD::createStencilMatrix(h.dms[1], stencil);
        PetscFD::assembleStencilOperator(h.dms[1], stencil, 1.0, {}, {}, B);

        // symbol of -Laplacian on axis d for the mode cos(2 pi k x / L), theta = 2 pi k / nc
        const auto symbol = [&](unsigned int d)
        {
          const double theta = 2.0 * pi * k[d] / nc;
          const double h2 = dx[d] * dx[d];
          return wide ? (2.5 - 8.0 / 3.0 * std::cos(theta) + 1.0 / 6.0 * std::cos(2.0 * theta)) / h2
                      : (2.0 - 2.0 * std::cos(theta)) / h2;
        };
        // fillVec gives unit coordinates per axis; 0.3 d is an arbitrary phase per axis
        const auto modes = [&](const std::array<double, 3> & x, bool scaled)
        {
          double v = 0.0;
          for (const auto d : make_range(dim))
            v += (scaled ? symbol(d) : 1.0) * std::cos(2.0 * pi * k[d] * x[d] + 0.3 * d);
          return v;
        };

        Vec u, Bu, expected;
        PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[1], &u));
        PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[1], &Bu));
        PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[1], &expected));
        fillVec(h.dms[1], u, [&](const std::array<double, 3> & x) { return modes(x, false); });
        fillVec(
            h.dms[1], expected, [&](const std::array<double, 3> & x) { return modes(x, true); });
        PetscCallAbort(PETSC_COMM_WORLD, MatMult(B, u, Bu));
        PetscReal scale;
        PetscCallAbort(PETSC_COMM_WORLD, VecNorm(expected, NORM_INFINITY, &scale));
        EXPECT_LT(maxDiff(Bu, expected), 1e-12 * scale) << "dim " << dim << " wide " << wide;

        PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&B));
        PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&u));
        PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&Bu));
        PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&expected));
      }
      MPI_Comm_free(&comm);
    }
}

// On the coarsest level (8 cells per axis) the 5-point biharmonic's offsets +4 and -4
// land on the same column; both must be added. A periodic mode is an eigenvector with
// eigenvalue the square of the 5-point Laplacian symbol.
TEST(PetscFDHierarchyTest, AssembledOperatorAddsWrappedOffsets)
{
  // each rank needs 4 coarse cells (the stencil width): at most 2 ranks
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size > 2)
    GTEST_SKIP() << "needs at most 2 ranks";

  const double pi = M_PI;
  const int64_t n = 16;
  const auto partition = uniformPartition(1, n);
  MPI_Comm comm = PetscFD::dmdaComm(partition);
  {
    PetscFD::Hierarchy h;
    PetscFD::buildHierarchy(partition, comm, 2, 4, false, {}, h);

    const int64_t nc = n / 2;
    const std::vector<Real> dx{1.0 / nc};
    const auto stencil = FiniteDifferenceBiharmonic::buildStencil(1, true, dx, 1.0);

    Mat B = PetscFD::createStencilMatrix(h.dms[1], stencil);
    PetscFD::assembleStencilOperator(h.dms[1], stencil, 1.0, {}, {}, B);

    for (const int k : {1, 2, 3, 4})
    {
      const double theta = 2.0 * pi * k / nc;
      const double symbol =
          (2.5 - 8.0 / 3.0 * std::cos(theta) + 1.0 / 6.0 * std::cos(2.0 * theta)) / (dx[0] * dx[0]);
      Vec u, Bu, expected;
      PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[1], &u));
      PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[1], &Bu));
      PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[1], &expected));
      const auto mode = [&](const std::array<double, 3> & x)
      { return std::cos(2.0 * pi * k * x[0] + 0.3); };
      fillVec(h.dms[1], u, mode);
      fillVec(h.dms[1],
              expected,
              [&](const std::array<double, 3> & x) { return symbol * symbol * mode(x); });
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(B, u, Bu));
      EXPECT_LT(maxDiff(Bu, expected), 1e-10 * symbol * symbol) << "k " << k;
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&u));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&Bu));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&expected));
    }
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&B));
  }
  MPI_Comm_free(&comm);
}

// The coarse matrices store only the stencil's columns, not the DMDA's (2w + 1)^dim box:
// 73 entries per row for the 5-point biharmonic in 3D on a box DMDA (729 in the box),
// and the assembled operator still maps a periodic mode to its symbol.
TEST(PetscFDHierarchyTest, StencilMatrixStoresOnlyTheStencil)
{
  // each rank needs 4 cells (the stencil width): at most 4 ranks on 16 cells per axis
  const double pi = M_PI;
  const int64_t n = 16;
  const auto partition = uniformPartition(3, n);
  MPI_Comm comm = PetscFD::dmdaComm(partition);
  {
    PetscFD::Hierarchy h;
    PetscFD::buildHierarchy(partition, comm, 1, 4, true, {}, h);

    const std::vector<Real> dx(3, 1.0 / n);
    const auto stencil = FiniteDifferenceBiharmonic::buildStencil(3, true, dx, 1.0);
    Mat B = PetscFD::createStencilMatrix(h.dms[0], stencil);
    PetscFD::assembleStencilOperator(h.dms[0], stencil, 1.0, {}, {}, B);

    MatInfo info;
    PetscInt rows;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetInfo(B, MAT_GLOBAL_SUM, &info));
    PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(B, &rows, nullptr));
    EXPECT_EQ(info.nz_allocated, 73.0 * rows);
    EXPECT_EQ(info.nz_used, 73.0 * rows);

    // cos(2 pi x_0 + 0.3): eigenvalue the squared axis-0 5-point Laplacian symbol
    const double theta = 2.0 * pi / n;
    const double symbol =
        (2.5 - 8.0 / 3.0 * std::cos(theta) + 1.0 / 6.0 * std::cos(2.0 * theta)) * n * n;
    Vec u, Bu, expected;
    PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[0], &u));
    PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[0], &Bu));
    PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[0], &expected));
    const auto mode = [&](const std::array<double, 3> & x)
    { return std::cos(2.0 * pi * x[0] + 0.3); };
    fillVec(h.dms[0], u, mode);
    fillVec(h.dms[0],
            expected,
            [&](const std::array<double, 3> & x) { return symbol * symbol * mode(x); });
    PetscCallAbort(PETSC_COMM_WORLD, MatMult(B, u, Bu));
    EXPECT_LT(maxDiff(Bu, expected), 1e-10 * symbol * symbol);
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&u));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&Bu));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&expected));
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&B));
  }
  MPI_Comm_free(&comm);
}

// With boundary faces, the coarse level is still cell-centred (n -> n/2) and the
// interpolation folds the neighbour outside a face onto the parent: with Neumann faces
// P reproduces constants (rows sum to 1), with Dirichlet faces it is exact next to the
// lower faces for multilinear fields that are odd about them.
TEST(PetscFDHierarchyTest, CellCentredInterpolationWithFaces)
{
  for (const unsigned int dim : {1u, 2u, 3u})
  {
    const int64_t n = dim == 3 ? 16 : 32;
    // Marlin axis 0 has faces; axis 1 periodic in 2D and 3D, axis 2 faces in 3D
    const auto partition = boundedPartition(dim, n, {false, true, false});
    MPI_Comm comm = PetscFD::dmdaComm(partition);
    for (const bool dirichlet : {false, true})
    {
      const auto faces = dirichlet ? uniformFaces(1.0, 0.0) : uniformFaces(0.0, 1.0);
      PetscFD::Hierarchy h;
      PetscFD::buildHierarchy(partition, comm, 2, 1, false, faces, h);
      PetscInt M;
      PetscCallAbort(PETSC_COMM_WORLD,
                     DMDAGetInfo(h.dms[1], nullptr, &M, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
      EXPECT_EQ(M, n / 2) << "dim " << dim;

      Vec coarse, fine, expected;
      PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[1], &coarse));
      PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[0], &fine));
      PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[0], &expected));
      if (!dirichlet)
      {
        PetscCallAbort(PETSC_COMM_WORLD, VecSet(coarse, 1.0));
        PetscCallAbort(PETSC_COMM_WORLD, VecSet(expected, 1.0));
      }
      else
      {
        // odd about the lower face of every axis with faces (x_0 and, in 3D, x_2), so the
        // coarse ghost -parent is its exact extension there; only the lower fine cells
        // are compared, since it is not odd about the upper faces
        const auto odd = [&](const std::array<double, 3> & x)
        { return 2.0 * x[0] * (dim == 3 ? 3.0 * x[2] : 1.0); };
        fillVec(h.dms[1], coarse, odd);
        fillVec(h.dms[0], expected, odd);
      }
      PetscCallAbort(PETSC_COMM_WORLD, MatMult(h.interpolations[0], coarse, fine));
      if (dirichlet)
      {
        // fine cells in the lower half of axis 0 (and of axis 2 in 3D): there the
        // interpolation is exact for a field linear in x_0 (bilinear with x_2)
        Vec mask;
        PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[0], &mask));
        fillVec(h.dms[0],
                mask,
                [&](const std::array<double, 3> & x)
                { return x[0] < 0.5 && (dim < 3 || x[2] < 0.5) ? 1.0 : 0.0; });
        PetscCallAbort(PETSC_COMM_WORLD, VecPointwiseMult(fine, fine, mask));
        PetscCallAbort(PETSC_COMM_WORLD, VecPointwiseMult(expected, expected, mask));
        PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&mask));
      }
      EXPECT_LT(maxDiff(fine, expected), 1e-13) << "dim " << dim << " dirichlet " << dirichlet;

      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&coarse));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&fine));
      PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&expected));
    }
    MPI_Comm_free(&comm);
  }
}

// The coarse operator with Dirichlet (Neumann) faces maps sin (cos) modes of the
// cell-centred grid to the periodic symbol: their odd (even) mirror is exactly the
// homogeneous ghost of every layer, for the 3- and the 5-point stencil.
TEST(PetscFDHierarchyTest, AssembledOperatorWithFaces)
{
  const double pi = M_PI;
  for (const unsigned int dim : {1u, 2u})
    for (const bool wide : {false, true})
      for (const bool dirichlet : {false, true})
      {
        const int64_t n = 32;
        // faces on Marlin axis 0, axis 1 periodic
        const auto partition = boundedPartition(dim, n, {false, true, true});
        MPI_Comm comm = PetscFD::dmdaComm(partition);
        {
          const auto faces = dirichlet ? uniformFaces(1.0, 0.0) : uniformFaces(0.0, 1.0);
          PetscFD::Hierarchy h;
          PetscFD::buildHierarchy(partition, comm, 2, wide ? 2 : 1, false, faces, h);

          const int64_t nc = n / 2;
          const std::vector<Real> dx(dim, 1.0 / nc);
          const auto stencil = FiniteDifferenceLaplacian::buildStencil(dim, wide, dx, -1.0);
          Mat B = PetscFD::createStencilMatrix(h.dms[1], stencil);
          PetscFD::assembleStencilOperator(h.dms[1], stencil, 1.0, faces, dx, B);

          // mode 3 on the face axis (sin for Dirichlet, cos for Neumann), mode 1 periodic
          const auto symbol = [&](double theta)
          {
            return (wide ? 2.5 - 8.0 / 3.0 * std::cos(theta) + 1.0 / 6.0 * std::cos(2.0 * theta)
                         : 2.0 - 2.0 * std::cos(theta)) *
                   nc * nc;
          };
          const double lambda = symbol(3.0 * pi / nc) + (dim > 1 ? symbol(2.0 * pi / nc) : 0.0);
          const auto mode = [&](const std::array<double, 3> & x)
          {
            const double face_mode =
                dirichlet ? std::sin(3.0 * pi * x[0]) : std::cos(3.0 * pi * x[0]);
            return face_mode * (dim > 1 ? std::cos(2.0 * pi * x[1] + 0.3) : 1.0);
          };
          Vec u, Bu, expected;
          PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[1], &u));
          PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[1], &Bu));
          PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(h.dms[1], &expected));
          fillVec(h.dms[1], u, mode);
          fillVec(h.dms[1],
                  expected,
                  [&](const std::array<double, 3> & x) { return lambda * mode(x); });
          PetscCallAbort(PETSC_COMM_WORLD, MatMult(B, u, Bu));
          EXPECT_LT(maxDiff(Bu, expected), 1e-11 * lambda)
              << "dim " << dim << " wide " << wide << " dirichlet " << dirichlet;

          PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&B));
          PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&u));
          PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&Bu));
          PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&expected));
        }
        MPI_Comm_free(&comm);
      }
}

// A Robin face changes only the diagonal of the 3-point operator next to it, by the
// ghost coefficient of the explicit ghost fill (RobinGhost::coefficientOnU0), and
// keeps it symmetric.
TEST(PetscFDHierarchyTest, AssembledOperatorWithRobinFace)
{
  const int64_t n = 16;
  const auto partition = boundedPartition(1, n, {false, true, true});
  MPI_Comm comm = PetscFD::dmdaComm(partition);
  {
    // lower face Robin a = 2, b = 0.3; upper face Dirichlet
    PetscFD::FaceGhosts faces;
    faces.ab[0] = {2.0, 0.3};
    faces.ab[1] = {1.0, 0.0};
    PetscFD::Hierarchy h;
    PetscFD::buildHierarchy(partition, comm, 1, 1, false, faces, h);
    const std::vector<Real> dx{1.0 / n};
    const auto stencil = FiniteDifferenceLaplacian::buildStencil(1, false, dx, -1.0);
    Mat B = PetscFD::createStencilMatrix(h.dms[0], stencil);
    PetscFD::assembleStencilOperator(h.dms[0], stencil, 1.0, faces, dx, B);

    PetscBool symmetric;
    PetscCallAbort(PETSC_COMM_WORLD, MatIsSymmetric(B, 0.0, &symmetric));
    EXPECT_TRUE(symmetric);

    // diagonal entries of the first and last row (each owned by one rank)
    PetscInt begin, end;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetOwnershipRange(B, &begin, &end));
    const double h2 = dx[0] * dx[0];
    for (const PetscInt row : {PetscInt(0), PetscInt(n - 1)})
      if (row >= begin && row < end)
      {
        PetscScalar value;
        PetscCallAbort(PETSC_COMM_WORLD, MatGetValues(B, 1, &row, 1, &row, &value));
        const double coeff = row == 0 ? RobinGhost::coefficientOnU0(2.0, 0.3, dx[0])
                                      : RobinGhost::coefficientOnU0(1.0, 0.0, dx[0]);
        EXPECT_NEAR(value, (2.0 - coeff) / h2, 1e-12 / h2) << "row " << row;
      }
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&B));
  }
  MPI_Comm_free(&comm);
}

// The Gershgorin bound of the smoother is the maximum over rows: on a periodic grid
// every row of the 5-point -Laplacian gives 1 + (2 (4/3 + 1/12)) / (5/2) = 32/15, the
// first row next to a Neumann face 1 + (5/4 + 1/12) / (5/2 - 4/3) = 15/7.
TEST(PetscFDHierarchyTest, JacobiSpectrumBoundWithFaces)
{
  const int64_t n = 16;
  const std::vector<Real> dx{1.0 / n};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(1, true, dx, -1.0);
  for (const bool faces_present : {false, true})
  {
    const auto partition = boundedPartition(1, n, {!faces_present, true, true});
    MPI_Comm comm = PetscFD::dmdaComm(partition);
    {
      const auto faces = uniformFaces(0.0, 1.0);
      PetscFD::Hierarchy h;
      PetscFD::buildHierarchy(partition, comm, 1, 2, false, faces, h);
      EXPECT_NEAR(PetscFD::jacobiSpectrumBound(h.dms[0], stencil, faces, dx),
                  faces_present ? 15.0 / 7.0 : 32.0 / 15.0,
                  1e-13);
    }
    MPI_Comm_free(&comm);
  }
}
