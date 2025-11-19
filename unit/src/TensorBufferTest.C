/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorBuffer.h"
#include "MarlinUtils.h"
#include "gtest/gtest.h"

#if 0
TEST(TensorBufferTest, Gradient)
{
  auto A = MooseTensor::createBuffer({20, 100}, {-pi, -pi}, {pi, 3 * pi});

  auto & a = A.data();
  auto [x, y] = A.getAxis();
  auto [i, j] = A.getFrequency();

  // function
  a = sin(2.0 * x) * sin(3.0 * y);

  // analytic derivatives
  auto dadx = 2.0 * cos(2.0 * x) * sin(3.0 * y);
  auto dady = sin(2.0 * x) * 3.0 * cos(3.0 * y);

  // spectral derivatives
  auto [grad_x, grad_y] = A.grad();

  // compute max difference of spectral and analytic derivatives
  EXPECT_NEAR((grad_x - dadx).abs().max().item<double>(), 0.0, 1e-12);
  EXPECT_NEAR((grad_y - dady).abs().max().item<double>(), 0.0, 1e-12);
}

TEST(TensorBufferTest, 2DAxis)
{
  auto A = MooseTensor::createBuffer({3, 4});
  A.min() = {0.0, 0.0};
  A.max() = {3.0 * 4.0 * 5.0, 4.0 * 5.0 * 6.0};

  const auto xCompare = [&A](auto interval, torch::detail::TensorDataContainer gold)
  {
    auto xt = A.getAxis(0, interval);
    auto xg = torch::unsqueeze(torch::tensor(gold), 1);
    xg = xg.to(xt.device());

    EXPECT_TRUE(torch::equal(xt, xg));
  };

  const auto yCompare = [&A](auto interval, torch::detail::TensorDataContainer gold)
  {
    auto yt = A.getAxis(1, interval);
    auto yg = torch::unsqueeze(torch::tensor(gold), 0);
    yg = yg.to(yt.device());

    EXPECT_TRUE(torch::equal(yt, yg));
  };

  xCompare(MooseTensor::Interval::CLOSED, {0.0, 30.0, 60.0});
  xCompare(MooseTensor::Interval::OPEN, {15.0, 30.0, 45.0});
  xCompare(MooseTensor::Interval::LEFT_OPEN, {20.0, 40.0, 60.0});
  xCompare(MooseTensor::Interval::RIGHT_OPEN, {0.0, 20.0, 40.0});

  yCompare(MooseTensor::Interval::CLOSED, {0.0, 40.0, 80.0, 120.0});
  yCompare(MooseTensor::Interval::OPEN, {24.0, 48.0, 72.0, 96.0});
  yCompare(MooseTensor::Interval::LEFT_OPEN, {30.0, 60.0, 90.0, 120.0});
  yCompare(MooseTensor::Interval::RIGHT_OPEN, {0.0, 30.0, 60.0, 90.0});
}

TEST(TensorBufferTest, 1DAxis)
{
  auto A = MooseTensor::createBuffer({4}, {0.0}, {4.0 * 5.0 * 6.0});

  const auto xCompare = [&A](auto interval, torch::detail::TensorDataContainer gold)
  {
    auto xt = A.getAxis(0, interval);
    auto xg = torch::unsqueeze(torch::tensor(gold), 0);
    xg = xg.to(xt.device());
    EXPECT_TRUE(torch::equal(xt, xg));
  };

  xCompare(MooseTensor::Interval::CLOSED, {0.0, 40.0, 80.0, 120.0});
  xCompare(MooseTensor::Interval::OPEN, {24.0, 48.0, 72.0, 96.0});
  xCompare(MooseTensor::Interval::LEFT_OPEN, {30.0, 60.0, 90.0, 120.0});
  xCompare(MooseTensor::Interval::RIGHT_OPEN, {0.0, 30.0, 60.0, 90.0});
}
#endif
