/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/**********************************************************************/

#include "GrainRemap.h"
#include "MarlinUtils.h"

#include "gtest/gtest.h"

#include <torch/torch.h>

namespace
{

const std::array<bool, 3> no_wrap{{false, false, false}};
const std::array<bool, 3> wrap_x{{false, true, false}}; // dim 1 is x in [ny, nx]
const std::array<bool, 3> wrap_xy{{true, true, false}};

torch::TensorOptions
floatOpts()
{
  return MooseTensor::floatTensorOptions();
}

torch::Tensor
makeBool(const std::vector<int64_t> & shape)
{
  return torch::zeros(shape, floatOpts().dtype(torch::kBool));
}

/// add a smooth (Gaussian) grain to channel `color` of eta [ny, nx, n_op]
void
addDisk(torch::Tensor & eta, float cy, float cx, float sigma, int color, bool wrap = false)
{
  const auto ny = eta.size(0);
  const auto nx = eta.size(1);
  auto y = torch::arange(ny, eta.options());
  auto x = torch::arange(nx, eta.options());
  auto grids = torch::meshgrid({y, x}, "ij");
  auto dy = torch::abs(grids[0] - cy);
  auto dx = torch::abs(grids[1] - cx);
  if (wrap)
  {
    dy = torch::minimum(dy, ny - dy);
    dx = torch::minimum(dx, nx - dx);
  }
  auto val = torch::exp(-(dy * dy + dx * dx) / (2.0f * sigma * sigma));
  auto current = eta.select(-1, color);
  eta.select(-1, color).copy_(torch::maximum(val, current));
}

} // namespace

TEST(GrainRemap, computeColorMasks2D)
{
  auto eta = torch::zeros({2, 2, 2}, floatOpts());
  eta.index_put_({0, 0, 1}, 0.2);
  eta.index_put_({0, 1, 0}, 0.3);
  eta.index_put_({1, 1, 0}, 0.5);
  eta.index_put_({1, 1, 1}, 0.4);

  auto masks = GrainRemap::computeColorMasks(eta, 0.05);
  ASSERT_EQ(masks.size(), 2u);

  EXPECT_FALSE(masks[0].index({0, 0}).item<bool>());
  EXPECT_TRUE(masks[1].index({0, 0}).item<bool>());
  EXPECT_TRUE(masks[0].index({0, 1}).item<bool>());
  EXPECT_FALSE(masks[1].index({0, 1}).item<bool>());
  // both zero, below threshold
  EXPECT_FALSE(masks[0].index({1, 0}).item<bool>());
  EXPECT_FALSE(masks[1].index({1, 0}).item<bool>());
  // color 0 is maximal
  EXPECT_TRUE(masks[0].index({1, 1}).item<bool>());
  EXPECT_FALSE(masks[1].index({1, 1}).item<bool>());
}

TEST(GrainRemap, labelConnectedComponents4vs8)
{
  auto mask = makeBool({2, 2});
  mask.index_put_({0, 0}, true);
  mask.index_put_({1, 1}, true); // diagonal separation

  auto labels4 = GrainRemap::labelConnectedComponents(mask, 4, no_wrap);
  EXPECT_EQ(labels4.index({0, 0}).item<int64_t>(), 0);
  EXPECT_EQ(labels4.index({1, 1}).item<int64_t>(), 1);
  EXPECT_EQ(labels4.index({0, 1}).item<int64_t>(), -1);

  auto labels8 = GrainRemap::labelConnectedComponents(mask, 8, no_wrap);
  EXPECT_EQ(labels8.index({0, 0}).item<int64_t>(), 0);
  EXPECT_EQ(labels8.index({1, 1}).item<int64_t>(), 0); // merged via the diagonal
}

TEST(GrainRemap, labelConnectedComponentsPeriodicWrap)
{
  // stripe crossing the x boundary
  auto mask = makeBool({8, 12});
  for (const int x : {0, 1, 10, 11})
    mask.index_put_({4, x}, true);

  auto labels = GrainRemap::labelConnectedComponents(mask, 4, no_wrap);
  EXPECT_NE(labels.index({4, 0}).item<int64_t>(), labels.index({4, 11}).item<int64_t>());

  auto labels_wrap = GrainRemap::labelConnectedComponents(mask, 4, wrap_x);
  EXPECT_EQ(labels_wrap.index({4, 0}).item<int64_t>(), labels_wrap.index({4, 11}).item<int64_t>());
  EXPECT_EQ(labels_wrap.index({4, 5}).item<int64_t>(), -1);
}

TEST(GrainRemap, expandLabels)
{
  auto labels = torch::full({9, 9}, -1, floatOpts().dtype(torch::kInt64));
  labels.index_put_({4, 4}, 7);

  auto expanded = GrainRemap::expandLabels(labels, 2, no_wrap);
  // Chebyshev ball of radius 2
  for (int64_t y = 0; y < 9; ++y)
    for (int64_t x = 0; x < 9; ++x)
    {
      const auto expected = (std::abs(y - 4) <= 2 && std::abs(x - 4) <= 2) ? 7 : -1;
      EXPECT_EQ(expanded.index({y, x}).item<int64_t>(), expected) << "at " << y << "," << x;
    }

  // expansion across a periodic boundary
  auto edge = torch::full({6, 6}, -1, floatOpts().dtype(torch::kInt64));
  edge.index_put_({3, 0}, 2);
  auto edge_nowrap = GrainRemap::expandLabels(edge, 1, no_wrap);
  EXPECT_EQ(edge_nowrap.index({3, 5}).item<int64_t>(), -1);
  auto edge_wrap = GrainRemap::expandLabels(edge, 1, wrap_x);
  EXPECT_EQ(edge_wrap.index({3, 5}).item<int64_t>(), 2);
}

TEST(GrainRemap, expandLabelsFirstArrivalOwnership)
{
  // a small-id grain must keep the cells it reaches first, even when a larger
  // id grain expands towards it later
  auto labels = torch::full({1, 9}, -1, floatOpts().dtype(torch::kInt64));
  labels.index_put_({0, 1}, 9); // large id left
  labels.index_put_({0, 7}, 3); // small id right

  auto expanded = GrainRemap::expandLabels(labels, 3, no_wrap);
  // cells closer to grain 3 belong to grain 3
  EXPECT_EQ(expanded.index({0, 6}).item<int64_t>(), 3);
  EXPECT_EQ(expanded.index({0, 5}).item<int64_t>(), 3);
  // cells closer to grain 9 belong to grain 9
  EXPECT_EQ(expanded.index({0, 2}).item<int64_t>(), 9);
  EXPECT_EQ(expanded.index({0, 3}).item<int64_t>(), 9);
  // the equidistant cell goes to the larger id
  EXPECT_EQ(expanded.index({0, 4}).item<int64_t>(), 9);
}

TEST(GrainRemap, componentMomentsBasic)
{
  // two rectangles on a 10x12 grid, no padding
  auto labels = torch::full({10, 12}, -1, floatOpts().dtype(torch::kInt64));
  labels.index_put_({torch::indexing::Slice(1, 4), torch::indexing::Slice(2, 6)}, 0);  // 3x4
  labels.index_put_({torch::indexing::Slice(6, 8), torch::indexing::Slice(8, 11)}, 1); // 2x3

  auto geom = GrainRemap::Geometry::serial(labels);
  auto moments = GrainRemap::computeComponentMoments(labels, 2, geom);
  ASSERT_EQ(moments.size(), 2u);

  EXPECT_DOUBLE_EQ(moments[0].volume, 12.0);
  EXPECT_NEAR(moments[0].sum[0] / moments[0].volume, 2.0, 1e-5); // y centroid
  EXPECT_NEAR(moments[0].sum[1] / moments[0].volume, 3.5, 1e-5); // x centroid
  EXPECT_DOUBLE_EQ(moments[1].volume, 6.0);
  EXPECT_NEAR(moments[1].sum[0] / moments[1].volume, 6.5, 1e-5);
  EXPECT_NEAR(moments[1].sum[1] / moments[1].volume, 9.0, 1e-5);
}

TEST(GrainRemap, componentMomentsCroppingAndOffset)
{
  // labels on a padded grid (pad=1, owned 4x4); the halo ring must not count
  auto labels = torch::full({6, 6}, -1, floatOpts().dtype(torch::kInt64));
  // fill everything with label 0, including the ring
  labels.fill_(0);

  GrainRemap::Geometry geom;
  geom.spatial_dim = 2;
  geom.pad = 1;
  geom.owned = {{4, 4, 0}};
  geom.global = {{16, 16, 0}};
  geom.global_begin = {{8, 4, 0}};

  auto moments = GrainRemap::computeComponentMoments(labels, 1, geom);
  ASSERT_EQ(moments.size(), 1u);
  EXPECT_DOUBLE_EQ(moments[0].volume, 16.0); // owned cells only
  // owned region covers global rows 8..11 and columns 4..7
  EXPECT_NEAR(moments[0].sum[0] / moments[0].volume, 9.5, 1e-5);
  EXPECT_NEAR(moments[0].sum[1] / moments[0].volume, 5.5, 1e-5);
}

TEST(GrainRemap, periodicCentroid)
{
  // component wrapping the x boundary: cells x = {46, 47, 0, 1} -> centroid 47.5
  auto labels = torch::full({8, 48}, -1, floatOpts().dtype(torch::kInt64));
  for (const int x : {46, 47, 0, 1})
    labels.index_put_({3, x}, 0);

  auto geom = GrainRemap::Geometry::serial(labels, wrap_x);
  auto moments = GrainRemap::computeComponentMoments(labels, 1, geom);
  auto grains = GrainRemap::finalizeGrains(moments, {0}, {0}, 1, geom);
  ASSERT_EQ(grains.size(), 1u);
  EXPECT_EQ(grains[0].volume, 4);
  EXPECT_NEAR(grains[0].centroid[0], 3.0, 1e-4);
  EXPECT_NEAR(grains[0].centroid[1], 47.5, 1e-3);
}

TEST(GrainRemap, seamEquivalencesAndMerge)
{
  // pad=1, owned 4x4. Label 5 owns the right edge and extends into the ghost
  // column; after the exchange the ghost column holds the neighbor's label 9.
  const auto int_opts = floatOpts().dtype(torch::kInt64);
  auto pre = torch::full({6, 6}, -1, int_opts);
  pre.index_put_({torch::indexing::Slice(1, 5), torch::indexing::Slice(3, 6)}, 5);
  auto post = pre.clone();
  post.index_put_({torch::indexing::Slice(1, 5), 5}, 9);

  GrainRemap::Geometry geom;
  geom.spatial_dim = 2;
  geom.pad = 1;
  geom.owned = {{4, 4, 0}};
  geom.global = {{4, 8, 0}};

  auto pairs = GrainRemap::detectSeamEquivalences(pre, post, geom);
  ASSERT_EQ(pairs.size(), 1u);
  EXPECT_EQ(pairs[0].first, 5);
  EXPECT_EQ(pairs[0].second, 9);

  int64_t n_grains = 0;
  auto label_to_grain = GrainRemap::mergeLabels(10, pairs, n_grains);
  EXPECT_EQ(n_grains, 9); // 10 labels, one merged pair
  EXPECT_EQ(label_to_grain[5], label_to_grain[9]);
  EXPECT_NE(label_to_grain[4], label_to_grain[5]);
}

TEST(GrainRemap, greedyRecolor)
{
  // path graph 0-1-2 with all-equal colors resolves with 2 colors
  std::vector<std::vector<int64_t>> path = {{1}, {0, 2}, {1}};
  std::vector<int> initial = {0, 0, 0};
  std::vector<int64_t> order = {0, 1, 2};
  std::vector<int64_t> conflicts;
  auto colors = GrainRemap::greedyRecolor(path, initial, 2, order, conflicts);
  EXPECT_TRUE(conflicts.empty());
  EXPECT_NE(colors[0], colors[1]);
  EXPECT_NE(colors[1], colors[2]);

  // triangle with 2 colors is infeasible and must report a conflict
  std::vector<std::vector<int64_t>> triangle = {{1, 2}, {0, 2}, {0, 1}};
  auto tri_colors = GrainRemap::greedyRecolor(triangle, initial, 2, order, conflicts);
  EXPECT_FALSE(conflicts.empty());

  // with 3 colors the triangle resolves
  auto tri3 = GrainRemap::greedyRecolor(triangle, initial, 3, order, conflicts);
  EXPECT_TRUE(conflicts.empty());
  EXPECT_NE(tri3[0], tri3[1]);
  EXPECT_NE(tri3[1], tri3[2]);
  EXPECT_NE(tri3[0], tri3[2]);
}

TEST(GrainRemap, remapOrderParametersPreservesUntouchedCells)
{
  const int64_t n = 16;
  auto eta = torch::zeros({n, n, 3}, floatOpts());
  addDisk(eta, 5, 5, 1.5, 0);
  addDisk(eta, 11, 11, 1.5, 2);
  auto eta_before = eta.clone();

  // move only the grain at (5,5): its dilated footprint is rows/cols 0..8
  auto changed = torch::full({n, n}, -1, floatOpts().dtype(torch::kInt64));
  changed.index_put_({torch::indexing::Slice(0, 9), torch::indexing::Slice(0, 9)}, 0);

  GrainRemap::remapOrderParameters(eta, changed, {0}, {1});

  // total mass conserved
  EXPECT_NEAR(eta.sum().item<float>(), eta_before.sum().item<float>(), 1e-4);
  // the other grain is untouched, including its sub-threshold tails
  auto far_region = [&](const torch::Tensor & t)
  { return t.index({torch::indexing::Slice(9, n), torch::indexing::Slice(9, n)}); };
  EXPECT_TRUE(torch::allclose(far_region(eta), far_region(eta_before)));
  // within the footprint the value moved from channel 0 to channel 1
  auto foot = [&](const torch::Tensor & t, int c)
  { return t.index({torch::indexing::Slice(0, 9), torch::indexing::Slice(0, 9), c}); };
  EXPECT_LT(foot(eta, 0).abs().max().item<float>(), 1e-6);
  EXPECT_TRUE(torch::allclose(foot(eta, 1), foot(eta_before, 0)));
}

TEST(GrainRemap, matchPersistentGrains)
{
  GrainRemap::Geometry geom;
  geom.spatial_dim = 2;
  geom.owned = {{48, 48, 0}};
  geom.global = {{48, 48, 0}};
  geom.periodic = {{false, true, false}};

  GrainRemap::GrainRemapOptions options;
  options.tracking_tolerance = 3.0;
  options.tracking_volume_ratio = 4.0;

  std::vector<GrainRemap::GrainMeta> previous(3);
  previous[0].persistent_id = 10;
  previous[0].centroid = {{10, 10, 0}};
  previous[0].volume = 100;
  previous[1].persistent_id = 11;
  previous[1].centroid = {{30, 30, 0}};
  previous[1].volume = 100;
  previous[2].persistent_id = 12;
  previous[2].centroid = {{20, 1, 0}}; // near the periodic x boundary
  previous[2].volume = 100;

  std::vector<GrainRemap::GrainMeta> current(3);
  current[0].centroid = {{31, 31, 0}}; // matches previous[1]
  current[0].volume = 110;
  current[1].centroid = {{20, 47, 0}}; // matches previous[2] across the wrap
  current[1].volume = 95;
  current[2].centroid = {{10.5, 10.5, 0}}; // near previous[0] but volume mismatch
  current[2].volume = 1000;

  auto ids = GrainRemap::matchPersistentGrains(previous, current, options, geom);
  EXPECT_EQ(ids[0], 11);
  EXPECT_EQ(ids[1], 12);
  EXPECT_EQ(ids[2], 13); // volume guard rejected the match -> new id
}

TEST(GrainRemap, persistentIdsOfVanishedGrainsAreNotReused)
{
  GrainRemap::Geometry geom;
  geom.spatial_dim = 2;
  geom.owned = {{48, 48, 0}};
  geom.global = {{48, 48, 0}};

  GrainRemap::GrainRemapOptions options;
  options.tracking_tolerance = 3.0;

  // grain 7 vanished in an earlier step; only grain 3 survives
  std::vector<GrainRemap::GrainMeta> previous(1);
  previous[0].persistent_id = 3;
  previous[0].centroid = {{10, 10, 0}};
  previous[0].volume = 100;

  std::vector<GrainRemap::GrainMeta> current(2);
  current[0].centroid = {{10, 10, 0}}; // matches previous grain 3
  current[0].volume = 100;
  current[1].centroid = {{40, 40, 0}}; // nucleated grain
  current[1].volume = 100;

  // without a monotone counter the new grain would reuse the vanished id 7+
  auto ids =
      GrainRemap::matchPersistentGrains(previous, current, options, geom, /*first_new_id=*/8);
  EXPECT_EQ(ids[0], 3);
  EXPECT_EQ(ids[1], 8);
}

TEST(GrainRemap, remapStepSeparatesCloseGrains)
{
  const int64_t n = 64;
  auto eta = torch::zeros({n, n, 2}, floatOpts());
  // two grains on color 0 close to each other, one far grain on color 1
  addDisk(eta, 32, 16, 2.0, 0);
  addDisk(eta, 32, 28, 2.0, 0);
  addDisk(eta, 10, 52, 2.0, 1);
  const auto total_before = eta.sum().item<float>();

  GrainRemap::GrainRemapOptions options;
  options.threshold = 0.1;
  options.halo_width = 4;

  auto res = GrainRemap::runRemapStep(eta, options, {});
  ASSERT_EQ(res.grains.size(), 3u);
  EXPECT_TRUE(res.conflicts.empty());
  ASSERT_EQ(res.remapped_grains.size(), 1u);

  // the two close grains must now occupy different order parameters
  std::vector<int> colors;
  for (const auto & g : res.grains)
    if (std::abs(g.centroid[0] - 32) < 2)
      colors.push_back(g.new_color);
  ASSERT_EQ(colors.size(), 2u);
  EXPECT_NE(colors[0], colors[1]);

  // mass conserved by the move
  EXPECT_NEAR(eta.sum().item<float>(), total_before, 1e-3);

  // a second step starting from the remapped field must be a no-op with stable ids
  auto res2 = GrainRemap::runRemapStep(eta, options, res.grains);
  ASSERT_EQ(res2.grains.size(), 3u);
  EXPECT_TRUE(res2.remapped_grains.empty());
  for (std::size_t i = 0; i < res2.grains.size(); ++i)
  {
    // match by centroid proximity
    bool found = false;
    for (std::size_t j = 0; j < res.grains.size(); ++j)
    {
      const double dy = res2.grains[i].centroid[0] - res.grains[j].centroid[0];
      const double dx = res2.grains[i].centroid[1] - res.grains[j].centroid[1];
      if (std::sqrt(dy * dy + dx * dx) < 1.0)
      {
        EXPECT_EQ(res2.grains[i].persistent_id, res.grains[j].persistent_id);
        EXPECT_EQ(res2.grains[i].color, res.grains[j].new_color);
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found);
  }
}

TEST(GrainRemap, remapStepFarGrainsShareOrderParameter)
{
  const int64_t n = 64;
  auto eta = torch::zeros({n, n, 2}, floatOpts());
  // two grains on color 0 far apart: no remap needed
  addDisk(eta, 16, 16, 2.0, 0);
  addDisk(eta, 48, 48, 2.0, 0);
  auto eta_before = eta.clone();

  GrainRemap::GrainRemapOptions options;
  options.threshold = 0.1;
  options.halo_width = 4;

  auto res = GrainRemap::runRemapStep(eta, options, {});
  ASSERT_EQ(res.grains.size(), 2u);
  EXPECT_TRUE(res.remapped_grains.empty());
  // the field is bit-identical (no destructive global rebuild)
  EXPECT_TRUE(torch::equal(eta, eta_before));
}

TEST(GrainRemap, remapStepPeriodicBoundary)
{
  const int64_t n = 48;
  GrainRemap::GrainRemapOptions options;
  options.threshold = 0.1;
  options.halo_width = 3;

  // two grains on color 0 adjacent only across the periodic x boundary
  auto make_eta = [&]()
  {
    auto eta = torch::zeros({n, n, 2}, floatOpts());
    addDisk(eta, 24, 4, 1.5, 0, /*wrap=*/true);
    addDisk(eta, 24, 44, 1.5, 0, /*wrap=*/true);
    return eta;
  };

  // without wrap they are far apart through the interior: no remap
  auto eta_nowrap = make_eta();
  auto res_nowrap = GrainRemap::runRemapStep(eta_nowrap, options, {});
  ASSERT_EQ(res_nowrap.grains.size(), 2u);
  EXPECT_TRUE(res_nowrap.remapped_grains.empty());

  // with periodic wrap they conflict and one moves to the free order parameter
  options.wrap = wrap_x;
  auto eta_wrap = make_eta();
  auto res_wrap = GrainRemap::runRemapStep(eta_wrap, options, {});
  ASSERT_EQ(res_wrap.grains.size(), 2u);
  ASSERT_EQ(res_wrap.remapped_grains.size(), 1u);
  EXPECT_NE(res_wrap.grains[0].new_color, res_wrap.grains[1].new_color);
}

TEST(GrainRemap, remapStepWrapSpanningGrain)
{
  const int64_t n = 48;
  GrainRemap::GrainRemapOptions options;
  options.threshold = 0.1;
  options.halo_width = 2;
  options.wrap = wrap_xy;

  // one grain centered on the periodic boundary must be detected as a single
  // grain with the correct wrapped centroid
  auto eta = torch::zeros({n, n, 2}, floatOpts());
  addDisk(eta, 24, 0, 2.0, 0, /*wrap=*/true);

  auto res = GrainRemap::runRemapStep(eta, options, {});
  ASSERT_EQ(res.grains.size(), 1u);
  EXPECT_NEAR(res.grains[0].centroid[0], 24.0, 0.1);
  const double cx = res.grains[0].centroid[1];
  EXPECT_TRUE(cx < 0.1 || cx > n - 0.1) << "wrapped centroid was " << cx;
}

TEST(GrainRemap, remapStep3D)
{
  const int64_t n = 32;
  auto eta = torch::zeros({n, n, n, 3}, floatOpts());

  auto addBlob = [&](float cz, float cy, float cx, float sigma, int color)
  {
    auto z = torch::arange(n, eta.options());
    auto grids = torch::meshgrid({z, z, z}, "ij");
    auto dz = grids[0] - cz;
    auto dy = grids[1] - cy;
    auto dx = grids[2] - cx;
    auto val = torch::exp(-(dz * dz + dy * dy + dx * dx) / (2.0f * sigma * sigma));
    eta.select(-1, color).copy_(torch::maximum(val, eta.select(-1, color)));
  };

  // two close blobs on color 0, far blobs on colors 1 and 2
  addBlob(16, 16, 10, 1.5, 0);
  addBlob(16, 16, 18, 1.5, 0);
  addBlob(8, 8, 26, 1.5, 1);
  addBlob(26, 26, 6, 1.5, 2);
  const auto total_before = eta.sum().item<float>();

  GrainRemap::GrainRemapOptions options;
  options.threshold = 0.1;
  options.halo_width = 3;
  options.connectivity = 26;

  auto res = GrainRemap::runRemapStep(eta, options, {});
  ASSERT_EQ(res.grains.size(), 4u);
  ASSERT_EQ(res.remapped_grains.size(), 1u);
  EXPECT_TRUE(res.conflicts.empty());
  EXPECT_NEAR(eta.sum().item<float>(), total_before, 1e-2);

  // the two close blobs now live on different order parameters
  std::vector<int> colors;
  for (const auto & g : res.grains)
    if (std::abs(g.centroid[0] - 16) < 2 && std::abs(g.centroid[1] - 16) < 2)
      colors.push_back(g.new_color);
  ASSERT_EQ(colors.size(), 2u);
  EXPECT_NE(colors[0], colors[1]);
}
