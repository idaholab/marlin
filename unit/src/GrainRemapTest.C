/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/**********************************************************************/

#include "GrainRemap.h"
#include "MarlinUtils.h"

#include "gtest/gtest.h"

namespace
{
const torch::TensorOptions float_opts = MooseTensor::floatTensorOptions();
const torch::TensorOptions int_opts = MooseTensor::intTensorOptions();

torch::Tensor makeFloat(const std::vector<int64_t> & shape)
{
  return torch::zeros(shape, float_opts);
}

torch::Tensor makeBool(const std::vector<int64_t> & shape)
{
  return torch::zeros(shape, float_opts.dtype(torch::kBool));
}

torch::Tensor makeInt(const std::vector<int64_t> & shape)
{
  return torch::zeros(shape, int_opts);
}
} // namespace

TEST(GrainRemap, computeColorMasks2D)
{
  auto eta = makeFloat({2, 2, 2});
  eta.index_put_({0, 0, 0}, 0.1);
  eta.index_put_({0, 0, 1}, 0.2);
  eta.index_put_({0, 1, 0}, 0.3);
  eta.index_put_({0, 1, 1}, 0.1);
  eta.index_put_({1, 0, 0}, 0.0);
  eta.index_put_({1, 0, 1}, 0.0);
  eta.index_put_({1, 1, 0}, 0.5);
  eta.index_put_({1, 1, 1}, 0.4);

  const double thresh = 0.05;
  auto masks = GrainRemap::computeColorMasks(eta, thresh);
  ASSERT_EQ(masks.size(), 2u);

  const auto flat0 = masks[0].flatten();
  const auto flat1 = masks[1].flatten();

  EXPECT_FALSE(flat0.index({0}).item<bool>()); // (0,0)
  EXPECT_TRUE(flat1.index({0}).item<bool>());
  EXPECT_TRUE(flat0.index({1}).item<bool>()); // (0,1)
  EXPECT_FALSE(flat1.index({1}).item<bool>());
  EXPECT_FALSE(flat0.index({2}).item<bool>()); // (1,0)
  EXPECT_FALSE(flat1.index({2}).item<bool>()); // both zero below thresh
  EXPECT_TRUE(flat0.index({3}).item<bool>());  // (1,1)
  EXPECT_FALSE(flat1.index({3}).item<bool>());
}

TEST(GrainRemap, labelConnectedComponents4vs8)
{
  auto mask = makeBool({2, 2});
  mask.index_put_({0, 0}, true);
  mask.index_put_({1, 1}, true); // diagonal separation

  GrainRemap::GrainRemapOptions opt4;
  opt4.connectivity = 4;
  auto labels4 = GrainRemap::labelConnectedComponents(mask, opt4);
  EXPECT_EQ(labels4.index({0, 0}).item<int64_t>(), 0);
  EXPECT_EQ(labels4.index({1, 1}).item<int64_t>(), 1);
  EXPECT_EQ(labels4.index({0, 1}).item<int64_t>(), -1);

  GrainRemap::GrainRemapOptions opt8;
  opt8.connectivity = 8;
  auto labels8 = GrainRemap::labelConnectedComponents(mask, opt8);
  EXPECT_EQ(labels8.index({0, 0}).item<int64_t>(), 0);
  EXPECT_EQ(labels8.index({1, 1}).item<int64_t>(), 0); // merged with diagonal neighbor
}

TEST(GrainRemap, remapOrderParameters)
{
  // 2x1 grid, two grains, two colors
  auto eta = makeFloat({2, 1, 2}); // shape [ny, nx, n_op]
  eta.index_put_({0, 0, 0}, 1.5); // grain 0, color 0
  eta.index_put_({1, 0, 1}, 2.5); // grain 1, color 1

  auto grain_ids = makeInt({2, 1});
  grain_ids.index_put_({0, 0}, 0);
  grain_ids.index_put_({1, 0}, 1);

  auto old_colors = torch::tensor({0, 1}, torch::TensorOptions().dtype(torch::kInt64));
  auto new_colors = torch::tensor({1, 0}, torch::TensorOptions().dtype(torch::kInt64));

  GrainRemap::remapOrderParameters(eta, grain_ids, old_colors, new_colors);

  EXPECT_NEAR(eta.index({0, 0, 1}).item<float>(), 1.5f, 1e-6);
  EXPECT_NEAR(eta.index({1, 0, 0}).item<float>(), 2.5f, 1e-6);
  EXPECT_NEAR(eta.index({0, 0, 0}).item<float>(), 0.0f, 1e-6);
  EXPECT_NEAR(eta.index({1, 0, 1}).item<float>(), 0.0f, 1e-6);
}

TEST(GrainRemap, colorAdjacencyWithPetsc)
{
  auto adj = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kInt64));
  adj.index_put_({0, 1}, 1);
  adj.index_put_({1, 0}, 1);
  adj.index_put_({1, 2}, 1);
  adj.index_put_({2, 1}, 1);

  auto colors = GrainRemap::colorAdjacencyWithPetsc(adj, /*n_colors=*/3, "power");
  ASSERT_EQ(colors.size(), 3u);
  EXPECT_NE(colors[0], colors[1]);
  EXPECT_NE(colors[1], colors[2]);
}
