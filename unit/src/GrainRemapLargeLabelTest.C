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

#include <cstdlib>
#include <string>

using torch::indexing::Slice;

TEST(GrainRemap, largeLabelDump)
{
  const int nz = 32, ny = 32, nx = 32;
  const int n_colors = 3;
  const double thresh = 0.1;

  auto eta = torch::zeros({nz, ny, nx, n_colors}, MooseTensor::floatTensorOptions());

  auto z = torch::arange(nz, eta.options());
  auto y = torch::arange(ny, eta.options());
  auto x = torch::arange(nx, eta.options());
  auto grids = torch::meshgrid({z, y, x}, "ij");
  auto Z = grids[0];
  auto Y = grids[1];
  auto X = grids[2];

  struct Blob
  {
    float cx, cy, cz, r, color;
  };
  std::vector<Blob> blobs = {{8.f, 8.f, 8.f, 6.f, 0.f},
                             {24.f, 16.f, 10.f, 7.f, 1.f},
                             {16.f, 24.f, 24.f, 6.f, 2.f}};

  for (const auto & b : blobs)
  {
    auto dz = Z - b.cz;
    auto dy = Y - b.cy;
    auto dx = X - b.cx;
    auto dist2 = dz * dz + dy * dy + dx * dx;
    auto mask = dist2 < (b.r * b.r);
    auto current = eta.select(-1, static_cast<int64_t>(b.color));
    auto updated = torch::where(mask, torch::ones_like(current), current);
    eta.index_put_({Slice(), Slice(), Slice(), static_cast<int64_t>(b.color)}, updated);
  }

  auto masks = GrainRemap::computeColorMasks(eta, thresh);
  ASSERT_EQ(masks.size(), static_cast<size_t>(n_colors));

  GrainRemap::GrainRemapOptions opt;
  opt.spatial_dim = 3;
  opt.connectivity = 26;
  opt.threshold = thresh;

  auto [raw_labels, compressed] = GrainRemap::labelConnectedComponentsWithRaw(masks[0], opt);

  EXPECT_EQ(raw_labels.sizes(), torch::IntArrayRef({nz, ny, nx}));
  EXPECT_EQ(compressed.sizes(), torch::IntArrayRef({nz, ny, nx}));
  EXPECT_GT(raw_labels.max().item<int64_t>(), 0);
  EXPECT_GE(compressed.max().item<int64_t>(), 0); // single component maps to label 0
  EXPECT_GT((compressed >= 0).sum().item<int64_t>(), 0);

  const char * env_path = std::getenv("GRAIN_LABEL_DUMP");
  const std::string out_path = env_path ? env_path : "grain_labels_raw.pt";
  try
  {
    torch::save(raw_labels.to(torch::Device(torch::kCPU)), out_path);
    std::cout << "Wrote raw labels (pre-compaction) to " << out_path << '\n';
  }
  catch (const std::exception & e)
  {
    ADD_FAILURE() << "Failed to save raw labels to " << out_path << ": " << e.what();
  }
}

TEST(GrainRemap, largeLabelDump2D)
{
  const int ny = 64, nx = 64;
  const int n_colors = 2;
  const double thresh = 0.1;

  auto eta = torch::zeros({ny, nx, n_colors}, MooseTensor::floatTensorOptions());
  auto y = torch::arange(ny, eta.options());
  auto x = torch::arange(nx, eta.options());
  auto grids = torch::meshgrid({y, x}, "ij");
  auto Y = grids[0];
  auto X = grids[1];

  struct Disk
  {
    float cx, cy, r, color;
  };
  // Multiple disks per color to verify distinct labels per color.
  std::vector<Disk> disks = {{16.f, 16.f, 8.f, 0.f},
                             {46.f, 42.f, 8.f, 0.f},
                             {20.f, 52.f, 6.f, 0.f},
                             {8.f, 40.f, 7.f, 1.f},
                             {40.f, 18.f, 7.f, 1.f},
                             {54.f, 20.f, 6.f, 1.f}};

  for (const auto & d : disks)
  {
    auto dy = Y - d.cy;
    auto dx = X - d.cx;
    auto dist2 = dy * dy + dx * dx;
    auto mask = dist2 < (d.r * d.r);
    auto current = eta.select(-1, static_cast<int64_t>(d.color));
    auto updated = torch::where(mask, torch::ones_like(current), current);
    eta.index_put_({Slice(), Slice(), static_cast<int64_t>(d.color)}, updated);
  }

  auto masks = GrainRemap::computeColorMasks(eta, thresh);
  ASSERT_EQ(masks.size(), static_cast<size_t>(n_colors));

  GrainRemap::GrainRemapOptions opt;
  opt.spatial_dim = 2;
  opt.connectivity = 8;
  opt.threshold = thresh;

  std::vector<torch::Tensor> raw_labels_vec;
  std::vector<torch::Tensor> compact_labels_vec;
  raw_labels_vec.reserve(n_colors);
  compact_labels_vec.reserve(n_colors);
  for (int c = 0; c < n_colors; ++c)
  {
    auto [raw_labels, compressed] =
        GrainRemap::labelConnectedComponentsWithRaw(masks[c], opt);
    EXPECT_GT(raw_labels.max().item<int64_t>(), 0);
    EXPECT_GE(compressed.max().item<int64_t>(), 0);
    EXPECT_GT((compressed >= 0).sum().item<int64_t>(), 0);
    auto compressed_cpu = compressed.to(torch::Device(torch::kCPU));
    auto uniq_tuple =
        torch::_unique(compressed_cpu.masked_select(compressed_cpu >= 0), /*sorted=*/true, /*return_inverse=*/false);
    auto uniq_vals = std::get<0>(uniq_tuple);
    EXPECT_GE(uniq_vals.numel(), 3); // three disks per color should be distinct labels
    raw_labels_vec.push_back(raw_labels.to(torch::Device(torch::kCPU)));
    compact_labels_vec.push_back(compressed.to(torch::Device(torch::kCPU)));
  }

  auto stacked = torch::stack(raw_labels_vec, 0); // shape [n_colors, ny, nx]

  std::vector<int64_t> offsets;
  auto combined = GrainRemap::buildGlobalContiguousLabels(compact_labels_vec, offsets);
  auto uniq_combined =
      torch::_unique(combined.masked_select(combined >= 0), /*sorted=*/true, /*return_inverse=*/false);
  EXPECT_GE(std::get<0>(uniq_combined).numel(), 6); // three disks per color -> at least 6 labels total

  const int halo = 1;
  auto halo_labels = GrainRemap::expandLabelsWithHalo(combined, halo);
  EXPECT_GT(halo_labels.max().item<int64_t>(), 0);

  auto adj_res = GrainRemap::buildHaloAdjacency(halo_labels, opt.connectivity);
  auto adjacency = adj_res.adjacency;
  auto uniq_vals = adj_res.unique_labels;
  const int64_t n_lbl = uniq_vals.numel();
  ASSERT_GT(n_lbl, 0);
  const auto * uniq_ptr = uniq_vals.data_ptr<int64_t>();

  // Color using PETSc MatColoring on the CPU adjacency.
  auto colors_vec = GrainRemap::colorAdjacencyWithPetsc(adjacency, /*n_colors=*/n_colors, "power");
  ASSERT_EQ(colors_vec.size(), static_cast<size_t>(n_lbl));
  std::vector<int64_t> colors_i64(colors_vec.begin(), colors_vec.end());
  auto colors_tensor = torch::from_blob(colors_i64.data(),
                                        {static_cast<long>(colors_i64.size())},
                                        torch::TensorOptions().dtype(torch::kInt64))
                           .clone();
  std::cout << "PETSc grain->op assignments (global_label -> color): ";
  for (int64_t i = 0; i < n_lbl; ++i)
  {
    std::cout << uniq_ptr[i] << "->" << colors_vec[i];
    if (i + 1 < n_lbl)
      std::cout << ", ";
  }
  std::cout << '\n';

  // Build old/new color lookup tables for remap.
  auto old_colors_t = GrainRemap::buildOldColorTable(compact_labels_vec, offsets, n_colors);
  auto new_colors_t = GrainRemap::buildNewColorTable(uniq_vals, colors_vec);

  // Construct a toy eta field: one-hot in the original color for each grain cell.
  auto eta_before = torch::zeros({ny, nx, n_colors}, MooseTensor::floatTensorOptions());
  auto eta_view = eta_before.view({-1, n_colors});
  auto combined_flat = combined.view({-1});
  auto old_color_flat = old_colors_t.index({combined_flat.clamp_min(0)});
  auto valid = combined_flat >= 0;
  auto rows = torch::arange(combined_flat.numel(), eta_view.options().dtype(torch::kInt64)).index({valid});
  auto cols = old_color_flat.index({valid});
  auto ones = torch::ones(rows.sizes(), eta_view.options());
  eta_view.index_put_({rows, cols}, ones);

  auto eta_after = eta_before.clone();
  auto combined_device = combined.to(eta_after.device());
  GrainRemap::remapOrderParameters(eta_after, combined_device, old_colors_t, new_colors_t);

  auto sum_before = eta_before.to(torch::kCPU).sum(0).sum(0);
  auto sum_after = eta_after.to(torch::kCPU).sum(0).sum(0);
  std::cout << "eta_before per-color sums: " << sum_before << '\n';
  std::cout << "eta_after  per-color sums: " << sum_after << '\n';

  auto color_grid = GrainRemap::buildLabelColorGrid(combined, uniq_vals, colors_vec);

  const char * env_path = std::getenv("GRAIN_LABEL_DUMP_2D");
  const std::string out_path = env_path ? env_path : "grain_labels_raw_2d.pt";
  const std::string out_combined = env_path ? (out_path + ".combined.pt") : "grain_labels_combined_2d.pt";
  const std::string out_halo = env_path ? (out_path + ".halo.pt") : "grain_labels_halo_2d.pt";
  const std::string out_adj = env_path ? (out_path + ".adjacency.pt") : "grain_labels_adjacency_2d.pt";
  const std::string out_colors = env_path ? (out_path + ".colors.pt") : "grain_colors_2d.pt";
  const std::string out_color_grid =
      env_path ? (out_path + ".colors_grid.pt") : "grain_colors_grid_2d.pt";
  const std::string out_eta_before =
      env_path ? (out_path + ".eta_before.pt") : "grain_eta_before_2d.pt";
  const std::string out_eta_after =
      env_path ? (out_path + ".eta_after.pt") : "grain_eta_after_2d.pt";
  try
  {
    torch::save(stacked, out_path);
    std::cout << "Wrote 2D raw labels (pre-compaction, per color) to " << out_path << '\n';
    torch::save(combined, out_combined);
    std::cout << "Wrote combined labels (all colors, pre-compaction) to " << out_combined << '\n';
    torch::save(halo_labels, out_halo);
    std::cout << "Wrote halo-expanded labels to " << out_halo << '\n';
    torch::save(adjacency, out_adj);
    std::cout << "Wrote halo-based adjacency matrix to " << out_adj << '\n';
    torch::save(colors_tensor, out_colors);
    std::cout << "Wrote PETSc-colored grain->op assignments to " << out_colors << '\n';
    torch::save(color_grid, out_color_grid);
    std::cout << "Wrote grain color grid to " << out_color_grid << '\n';
    torch::save(eta_before.to(torch::kCPU), out_eta_before);
    std::cout << "Wrote eta_before (original per-color fields) to " << out_eta_before << '\n';
    torch::save(eta_after.to(torch::kCPU), out_eta_after);
    std::cout << "Wrote eta_after  (remapped per-color fields) to " << out_eta_after << '\n';
  }
  catch (const std::exception & e)
  {
    ADD_FAILURE() << "Failed to save 2D raw labels to " << out_path << ": " << e.what();
  }
}
