/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FiniteDifferenceLaplacian.h"
#include "FiniteDifferenceBiharmonic.h"
#include "gtest/gtest.h"

#include "libmesh/int_range.h"

#include <map>
#include <vector>

namespace
{
// Sum of stencilAt()'s coefficients applied to u(x) = x^2 (per axis, isotropic)
// against the known analytic Laplacian 2*dim, as an independent sanity check
// that stencilAt() produces a consistent discrete Laplacian: for u = sum_axis
// x_axis^2 sampled at offsets {-2,-1,0,1,2}*dx from a point, each axis'
// stencil row applied to x^2 recovers exactly 2 (to machine precision for the
// 3-pt stencil, and to O(dx^4) truncation, i.e. exactly for a pure quadratic,
// for the 5-pt stencil), independent of the convolution-kernel code path in
// computeBuffer().
Real
applyStencilToQuadratic(const std::vector<FDStencilEntry> & stencil, const std::vector<Real> & dx)
{
  Real total = 0.0;
  for (const auto & entry : stencil)
  {
    Real x = 0.0;
    for (std::size_t axis = 0; axis < entry.offset.size(); ++axis)
      x += entry.offset[axis] * dx[axis];
    total += entry.coefficient * x * x;
  }
  return total;
}
}

TEST(LinearFDOperatorTest, Stencil3pt1D)
{
  const std::vector<Real> dx{0.1};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(1, false, dx, 1.0);

  ASSERT_EQ(stencil.size(), 3u);
  EXPECT_NEAR(applyStencilToQuadratic(stencil, dx), 2.0, 1e-10);
}

TEST(LinearFDOperatorTest, Stencil5pt1D)
{
  const std::vector<Real> dx{0.1};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(1, true, dx, 1.0);

  ASSERT_EQ(stencil.size(), 5u);
  EXPECT_NEAR(applyStencilToQuadratic(stencil, dx), 2.0, 1e-10);
}

TEST(LinearFDOperatorTest, Stencil3pt2D)
{
  const std::vector<Real> dx{0.2, 0.3};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(2, false, dx, 1.0);

  ASSERT_EQ(stencil.size(), 6u);
  EXPECT_NEAR(applyStencilToQuadratic(stencil, dx), 4.0, 1e-10);
}

TEST(LinearFDOperatorTest, Stencil5pt3D)
{
  const std::vector<Real> dx{0.15, 0.25, 0.35};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(3, true, dx, 1.0);

  ASSERT_EQ(stencil.size(), 15u);
  EXPECT_NEAR(applyStencilToQuadratic(stencil, dx), 6.0, 1e-10);
}

TEST(LinearFDOperatorTest, StencilAppliesFactor)
{
  const std::vector<Real> dx{0.1};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(1, false, dx, 2.5);

  EXPECT_NEAR(applyStencilToQuadratic(stencil, dx), 2.0 * 2.5, 1e-10);
}

// Direct coefficient-for-coefficient check against the literal values
// computeBuffer() uses to build its convolution kernel (see
// FiniteDifferenceLaplacian.C), so the two cannot silently diverge.
TEST(LinearFDOperatorTest, Stencil3ptMatchesConvKernelCoefficients)
{
  const std::vector<Real> dx{1.0};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(1, false, dx, 1.0);

  std::map<int, Real> by_offset;
  for (const auto & entry : stencil)
    by_offset[entry.offset[0]] = entry.coefficient;

  EXPECT_NEAR(by_offset.at(0), -2.0, 1e-12);
  EXPECT_NEAR(by_offset.at(1), 1.0, 1e-12);
  EXPECT_NEAR(by_offset.at(-1), 1.0, 1e-12);
}

TEST(LinearFDOperatorTest, Stencil5ptMatchesConvKernelCoefficients)
{
  const std::vector<Real> dx{1.0};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(1, true, dx, 1.0);

  std::map<int, Real> by_offset;
  for (const auto & entry : stencil)
    by_offset[entry.offset[0]] = entry.coefficient;

  EXPECT_NEAR(by_offset.at(0), -2.5, 1e-12);
  EXPECT_NEAR(by_offset.at(1), 4.0 / 3.0, 1e-12);
  EXPECT_NEAR(by_offset.at(-1), 4.0 / 3.0, 1e-12);
  EXPECT_NEAR(by_offset.at(2), -1.0 / 12.0, 1e-12);
  EXPECT_NEAR(by_offset.at(-2), -1.0 / 12.0, 1e-12);
}

// Coarse multigrid levels rediscretise at spacing 2h: every Laplacian
// coefficient scales as 1/dx^2, so doubling the spacing divides it by 4.
TEST(LinearFDOperatorTest, StencilRediscretisedAtDoubleSpacing)
{
  for (const bool wide : {false, true})
  {
    const std::vector<Real> dx{0.1, 0.3};
    const std::vector<Real> dx2{0.2, 0.6};
    const auto fine = FiniteDifferenceLaplacian::buildStencil(2, wide, dx, -1.5);
    const auto coarse = FiniteDifferenceLaplacian::buildStencil(2, wide, dx2, -1.5);

    ASSERT_EQ(fine.size(), coarse.size());
    for (const auto i : make_range(fine.size()))
    {
      EXPECT_EQ(fine[i].offset, coarse[i].offset);
      EXPECT_NEAR(coarse[i].coefficient, fine[i].coefficient / 4.0, 1e-10);
    }
  }
}

// The composed biharmonic stencil applied to u = |x|^4 at x = 0 must give the
// analytic Laplacian(Laplacian(u)) = 8 dim (dim + 2). Both Laplacian stencils
// reproduce it exactly: the 3-point one maps x^4 to 12 x^2 + 2 h^2, whose
// constant the second application removes.
TEST(LinearFDOperatorTest, BiharmonicStencilOnQuartic)
{
  for (const bool wide : {false, true})
    for (const unsigned int dim : {1u, 2u, 3u})
    {
      const std::vector<Real> dx{0.1, 0.2, 0.3};
      const std::vector<Real> h(dx.begin(), dx.begin() + dim);
      const auto stencil = FiniteDifferenceBiharmonic::buildStencil(dim, wide, h, 1.0);
      Real total = 0.0;
      for (const auto & entry : stencil)
      {
        Real r2 = 0.0;
        for (const auto d : make_range(dim))
          r2 += entry.offset[d] * h[d] * entry.offset[d] * h[d];
        total += entry.coefficient * r2 * r2;
      }
      EXPECT_NEAR(total, 8.0 * dim * (dim + 2), 1e-8) << "dim " << dim << " wide " << wide;
    }
}

// 3-point composed stencil in 2D at h = 1: the 13-point stencil (20 centre, -8 axis
// neighbours, 2 diagonals, 1 at distance 2 on the axes); one entry per offset.
TEST(LinearFDOperatorTest, Biharmonic13PointStencil)
{
  const auto stencil = FiniteDifferenceBiharmonic::buildStencil(2, false, {1.0, 1.0}, 1.0);
  ASSERT_EQ(stencil.size(), 13u);
  std::map<std::vector<int>, Real> by_offset;
  for (const auto & entry : stencil)
    by_offset[entry.offset] = entry.coefficient;
  EXPECT_NEAR(by_offset.at({0, 0}), 20.0, 1e-12);
  EXPECT_NEAR(by_offset.at({1, 0}), -8.0, 1e-12);
  EXPECT_NEAR(by_offset.at({0, -1}), -8.0, 1e-12);
  EXPECT_NEAR(by_offset.at({1, 1}), 2.0, 1e-12);
  EXPECT_NEAR(by_offset.at({-1, 1}), 2.0, 1e-12);
  EXPECT_NEAR(by_offset.at({2, 0}), 1.0, 1e-12);
  EXPECT_NEAR(by_offset.at({0, -2}), 1.0, 1e-12);
}

// Coarse levels rediscretise at 2h: every biharmonic coefficient scales as 1/h^4.
TEST(LinearFDOperatorTest, BiharmonicRediscretisedAtDoubleSpacing)
{
  const auto fine = FiniteDifferenceBiharmonic::buildStencil(2, true, {0.1, 0.3}, 2.0);
  const auto coarse = FiniteDifferenceBiharmonic::buildStencil(2, true, {0.2, 0.6}, 2.0);
  ASSERT_EQ(fine.size(), coarse.size());
  for (const auto i : make_range(fine.size()))
  {
    EXPECT_EQ(fine[i].offset, coarse[i].offset);
    EXPECT_NEAR(coarse[i].coefficient, fine[i].coefficient / 16.0, 1e-8);
  }
}
