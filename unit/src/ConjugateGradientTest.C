/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MarlinUtils.h"
#include "gtest/gtest.h"

TEST(ConjugateGradientTest, solve2d)
{
  const auto A = torch::tensor({{4.0, 1.0}, {1.0, 3.0}});
  const auto b = torch::tensor({1.0, 2.0});
  auto Afunc = [&A](const torch::Tensor & x) { return A.matmul(x); };

  const auto [x, it, norm] = MooseTensor::conjugateGradientSolve(Afunc, b);

  EXPECT_EQ(it, 2);
  EXPECT_NEAR(norm, 0, 1e-9);
  EXPECT_NEAR((Afunc(x) - b).norm().item<double>(), 0.0, 1e-9);
}

TEST(ConjugateGradientTest, solve4d)
{
  const auto A = torch::tensor(
      {{4.0, 1.0, 2.0, 3.0}, {1.0, 5.0, 1.0, 2.0}, {2.0, 1.0, 6.0, 1.0}, {3.0, 2.0, 1.0, 7.0}});
  const auto b = torch::tensor({1.0, 2.0, 3.0, 4.0});
  auto Afunc = [&A](const torch::Tensor & x) { return A.matmul(x); };

  const auto [x, it, norm] = MooseTensor::conjugateGradientSolve(Afunc, b);

  EXPECT_EQ(it, 4);
  EXPECT_NEAR(norm, 0, 1e-6);
  EXPECT_NEAR((Afunc(x) - b).norm().item<double>(), 0.0, 1e-6);
}
