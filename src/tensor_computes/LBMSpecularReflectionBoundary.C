/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMSpecularReflectionBoundary.h"
#include "LBMBoundaryCondition.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

#include <unordered_map>

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMSpecularReflectionBoundary);

namespace
{

/**
 * 52 known D2Q9 boundary types, identified by their decimal encoding of the
 * 9-bit connectivity pattern (reordered as {7,3,6,4,0,2,8,1,5}).
 */
constexpr int N_BND_TYPES = 52;

constexpr int BND_TYPES[N_BND_TYPES] = {
    31,  55,  63,  91,  95,  119, 127, 217, 219, 221, 223, 247, 253, 255, 287, 310, 311, 319,
    347, 351, 374, 375, 379, 382, 383, 415, 436, 437, 438, 439, 445, 447, 472, 473, 475, 476,
    477, 478, 479, 496, 497, 499, 500, 501, 502, 503, 504, 505, 507, 508, 509, 510};

/**
 * Specular reflection direction lookup table: icsr[bnd_type][direction].
 * For each of the 52 boundary types and each of the 9 D2Q9 directions,
 * gives the lattice direction index the reflected distribution should go to.
 * Values are 1-based (MATLAB convention)
 * Shape: 52 x 9, stored row-major.
 */
constexpr int ICSR[N_BND_TYPES][9] = {
    {1, 2, 3, 3, 2, 6, 7, 6, 9}, {1, 2, 2, 5, 5, 6, 9, 8, 9}, {1, 2, 3, 2, 5, 6, 6, 9, 9},
    {1, 2, 3, 3, 2, 6, 7, 6, 9}, {1, 2, 3, 3, 2, 6, 7, 6, 9}, {1, 2, 2, 5, 5, 6, 9, 8, 9},
    {1, 2, 3, 2, 5, 6, 6, 9, 9}, {1, 3, 3, 4, 4, 6, 7, 8, 7}, {1, 2, 3, 4, 3, 6, 7, 7, 6},
    {1, 3, 3, 4, 4, 6, 7, 8, 7}, {1, 2, 3, 4, 3, 6, 7, 7, 6}, {1, 2, 9, 4, 5, 2, 5, 4, 9},
    {1, 7, 3, 4, 5, 3, 7, 5, 4}, {1, 2, 3, 4, 5, 6, 7, 6, 9}, {1, 2, 3, 3, 2, 6, 7, 6, 9},
    {1, 2, 2, 5, 5, 6, 9, 8, 9}, {1, 2, 2, 5, 5, 6, 9, 8, 9}, {1, 2, 3, 2, 5, 6, 6, 9, 9},
    {1, 2, 3, 3, 2, 6, 7, 6, 9}, {1, 2, 3, 3, 2, 6, 7, 6, 9}, {1, 2, 2, 5, 5, 6, 9, 8, 9},
    {1, 2, 2, 5, 5, 6, 9, 8, 9}, {1, 2, 3, 6, 5, 6, 3, 2, 5}, {1, 2, 3, 9, 5, 3, 2, 5, 9},
    {1, 2, 3, 2, 5, 6, 6, 9, 9}, {1, 2, 3, 4, 6, 6, 4, 3, 2}, {1, 5, 4, 4, 5, 8, 7, 8, 9},
    {1, 5, 4, 4, 5, 8, 7, 8, 9}, {1, 2, 5, 4, 5, 9, 8, 8, 9}, {1, 2, 5, 4, 5, 9, 8, 8, 9},
    {1, 8, 3, 4, 5, 4, 3, 8, 5}, {1, 2, 3, 4, 5, 6, 9, 8, 9}, {1, 3, 3, 4, 4, 6, 7, 8, 7},
    {1, 3, 3, 4, 4, 6, 7, 8, 7}, {1, 2, 3, 4, 3, 6, 7, 7, 6}, {1, 3, 3, 4, 4, 6, 7, 8, 7},
    {1, 3, 3, 4, 4, 6, 7, 8, 7}, {1, 2, 3, 4, 7, 2, 7, 4, 3}, {1, 2, 3, 4, 3, 6, 7, 7, 6},
    {1, 5, 4, 4, 5, 8, 7, 8, 9}, {1, 5, 4, 4, 5, 8, 7, 8, 9}, {1, 2, 8, 4, 5, 5, 4, 8, 2},
    {1, 5, 4, 4, 5, 8, 7, 8, 9}, {1, 5, 4, 4, 5, 8, 7, 8, 9}, {1, 2, 5, 4, 5, 9, 8, 8, 9},
    {1, 2, 5, 4, 5, 9, 8, 8, 9}, {1, 4, 3, 4, 5, 7, 7, 8, 8}, {1, 4, 3, 4, 5, 7, 7, 8, 8},
    {1, 2, 3, 4, 5, 6, 7, 8, 7}, {1, 4, 3, 4, 5, 7, 7, 8, 8}, {1, 4, 3, 4, 5, 7, 7, 8, 8},
    {1, 2, 3, 4, 5, 8, 7, 8, 9}};

/**
 * Streaming permission lookup: ifstream[bnd_type][direction].
 * 1 = streaming is allowed in that direction, 0 = boundary treatment needed.
 * Shape: 52 x 9, stored row-major.
 */
constexpr int IF_STREAM[N_BND_TYPES][9] = {
    {1, 1, 1, 0, 0, 1, 0, 0, 0}, {1, 1, 0, 0, 1, 0, 0, 0, 1}, {1, 1, 1, 0, 1, 1, 0, 0, 1},
    {1, 1, 1, 0, 0, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 1, 0, 0, 0}, {1, 1, 0, 0, 1, 0, 0, 0, 1},
    {1, 1, 1, 0, 1, 1, 0, 0, 1}, {1, 0, 1, 1, 0, 0, 1, 0, 0}, {1, 1, 1, 1, 0, 1, 1, 0, 0},
    {1, 0, 1, 1, 0, 0, 1, 0, 0}, {1, 1, 1, 1, 0, 1, 1, 0, 0}, {1, 1, 0, 1, 1, 0, 0, 0, 1},
    {1, 0, 1, 1, 1, 0, 1, 0, 0}, {1, 1, 1, 1, 1, 1, 1, 0, 1}, {1, 1, 1, 0, 0, 1, 0, 0, 0},
    {1, 1, 0, 0, 1, 0, 0, 0, 1}, {1, 1, 0, 0, 1, 0, 0, 0, 1}, {1, 1, 1, 0, 1, 1, 0, 0, 1},
    {1, 1, 1, 0, 0, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 1, 0, 0, 0}, {1, 1, 0, 0, 1, 0, 0, 0, 1},
    {1, 1, 0, 0, 1, 0, 0, 0, 1}, {1, 1, 1, 0, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 1, 0, 0, 0, 1},
    {1, 1, 1, 0, 1, 1, 0, 0, 1}, {1, 1, 1, 1, 0, 1, 0, 0, 0}, {1, 0, 0, 1, 1, 0, 0, 1, 0},
    {1, 0, 0, 1, 1, 0, 0, 1, 0}, {1, 1, 0, 1, 1, 0, 0, 1, 1}, {1, 1, 0, 1, 1, 0, 0, 1, 1},
    {1, 0, 1, 1, 1, 0, 0, 1, 0}, {1, 1, 1, 1, 1, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 0, 1, 0, 0},
    {1, 0, 1, 1, 0, 0, 1, 0, 0}, {1, 1, 1, 1, 0, 1, 1, 0, 0}, {1, 0, 1, 1, 0, 0, 1, 0, 0},
    {1, 0, 1, 1, 0, 0, 1, 0, 0}, {1, 1, 1, 1, 0, 0, 1, 0, 0}, {1, 1, 1, 1, 0, 1, 1, 0, 0},
    {1, 0, 0, 1, 1, 0, 0, 1, 0}, {1, 0, 0, 1, 1, 0, 0, 1, 0}, {1, 1, 0, 1, 1, 0, 0, 1, 0},
    {1, 0, 0, 1, 1, 0, 0, 1, 0}, {1, 0, 0, 1, 1, 0, 0, 1, 0}, {1, 1, 0, 1, 1, 0, 0, 1, 1},
    {1, 1, 0, 1, 1, 0, 0, 1, 1}, {1, 0, 1, 1, 1, 0, 1, 1, 0}, {1, 0, 1, 1, 1, 0, 1, 1, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 0}, {1, 0, 1, 1, 1, 0, 1, 1, 0}, {1, 0, 1, 1, 1, 0, 1, 1, 0},
    {1, 1, 1, 1, 1, 0, 1, 1, 1}};

/**
 * Reorder indices to convert from D2Q9 direction order to the binary encoding
 * order used by the boundary type classification.
 * d2q9_order_to_new_order = {7, 3, 6, 4, 0, 2, 8, 1, 5}
 */
constexpr int D2Q9_TO_ENCODE_ORDER[9] = {7, 3, 6, 4, 0, 2, 8, 1, 5};

} // anonymous namespace

InputParameters
LBMSpecularReflectionBoundary::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addClassDescription(
      "LBM combination of bounce-back and specular reflection boundary condition. "
      "Uses a precomputed lookup table to determine specular reflection directions "
      "for each boundary node type in complex geometries (D2Q9 only).");
  params.addRequiredParam<TensorInputBufferName>("f_old", "Old timestep distribution function");
  params.addRequiredParam<TensorInputBufferName>("local_Knudsen_number", "Local Knudsen number");
  return params;
}

LBMSpecularReflectionBoundary::LBMSpecularReflectionBoundary(const InputParameters & parameters)
  : LBMBoundaryCondition(parameters),
    _f_old(_lb_problem.getBufferOld(getParam<TensorInputBufferName>("f_old"), 1, _radius)),
    _local_Knudsen_number(getInputBuffer("local_Knudsen_number", _radius)),
    _indices_built(false)
{
  if (_stencil._q != 9)
    mooseError("LBMSpecularReflectionBoundary currently only supports D2Q9 stencils.");

  if (_lb_problem.isBinaryMedia())
    maskBoundary();
  else
    mooseError(
        "LBMSpecularReflectionBoundary requires a binary media buffer to identify boundary nodes.");
}

void
LBMSpecularReflectionBoundary::buildSpecularIndices()
{
  // Classify each fluid node adjacent to a solid: determine its boundary type
  // by encoding which of its 9 neighbors are fluid (1) or solid/missing (0).
  // The encoding uses a reordering: direction order {7,3,6,4,0,2,8,1,5}
  // to form a 9-bit number. The result is looked up in BND_TYPES[].
  const int64_t nx_part = _binary_mesh.size(0);
  const int64_t ny_part = _binary_mesh.size(1);

  // map from decimal boundary type to its index in BND_TYPES
  std::unordered_map<int, int> bnd_type_map;
  for (int b = 0; b < N_BND_TYPES; b++)
    bnd_type_map[BND_TYPES[b]] = b;

  // boundary entries: (x, y, z, incoming_dir, specular_dir)
  std::vector<int64_t> entry_x, entry_y, entry_z, entry_ic, entry_sr;

  // Work on CPU for accessor compatibility (device-portable)
  auto mesh_cpu = _binary_mesh.cpu();
  auto mesh_accessor = mesh_cpu.accessor<int64_t, 3>();
  int64_t k = 0;
  for (int64_t i = _radius; i < nx_part - _radius; i++)
  {
    for (int64_t j = _radius; j < ny_part - _radius; j++)
    {
      if (mesh_accessor[i][j][k] != -1)
        continue;

      // 9-bit connectivity string in the reordered direction order
      int decimal_code = 0;
      for (int bit = 0; bit < 9; bit++)
      {
        int ic = D2Q9_TO_ENCODE_ORDER[bit];
        int64_t ni = i + _stencil._ex[ic].item<int64_t>();
        int64_t nj = j + _stencil._ey[ic].item<int64_t>();

        // ensure preiodicity in y direction
        nj = (nj <= -1) ? nj + ny_part : nj;
        nj = (nj >= ny_part) ? nj - ny_part : nj;

        // ensure preiodicity in x direction
        ni = (ni <= -1) ? ni + nx_part : ni;
        ni = (ni >= nx_part) ? ni - nx_part : ni;

        bool has_fluid_neighbor = false;
        if (ni >= 0 && ni < nx_part && nj >= 0 && nj < ny_part)
        {
          int neighbor_val = mesh_accessor[ni][nj][k];
          has_fluid_neighbor = (neighbor_val == 0);
        }

        if (!has_fluid_neighbor)
          decimal_code |= (1 << (8 - bit));
      }

      // interior nodes
      if (decimal_code == 511)
        continue;

      // boundary type
      auto it = bnd_type_map.find(decimal_code);
      if (it == bnd_type_map.end())
        mooseError("Encountered unknown boundary type with code " + std::to_string(decimal_code) +
                   ". This likely indicates a node connectivity pattern not covered by the current "
                   "lookup table.");

      int bnd_idx = it->second;

      // For each direction where streaming is NOT allowed, register a boundary entry
      for (int ic = 1; ic < 9; ic++)
      {
        if (IF_STREAM[bnd_idx][ic] == 0)
        {
          // ICSR values are 1-based, convert to 0-based
          int sr_dir = ICSR[bnd_idx][ic] - 1;
          entry_x.push_back(i);
          entry_y.push_back(j);
          entry_z.push_back(k);
          entry_ic.push_back(ic);
          entry_sr.push_back(sr_dir);
        }
      }
    }
  }
  const int64_t n_entries = static_cast<int64_t>(entry_x.size());

  if (n_entries > 0)
  {
    // Build index tensors on CPU, then transfer to configured device
    auto cpu_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto be_cpu = torch::zeros({n_entries, 4}, cpu_opts);
    auto sd_cpu = torch::zeros({n_entries}, cpu_opts);

    auto be_acc = be_cpu.accessor<int64_t, 2>();
    auto sd_acc = sd_cpu.accessor<int64_t, 1>();

    for (int64_t e = 0; e < n_entries; e++)
    {
      be_acc[e][0] = entry_x[e];
      be_acc[e][1] = entry_y[e];
      be_acc[e][2] = entry_z[e];
      be_acc[e][3] = entry_ic[e];
      sd_acc[e] = entry_sr[e];
    }

    _boundary_entries = be_cpu.to(MooseTensor::intTensorOptions());
    _specular_directions = sd_cpu.to(MooseTensor::intTensorOptions());
  }
  else
  {
    _boundary_entries = torch::zeros({0, 4}, MooseTensor::intTensorOptions());
    _specular_directions = torch::zeros({0}, MooseTensor::intTensorOptions());
  }
}

void
LBMSpecularReflectionBoundary::wallBoundary()
{
  if (!_indices_built)
  {
    buildSpecularIndices();
    _indices_built = true;

    if (_boundary_entries.size(0) == 0)
      mooseError(
          "LBMSpecularReflectionBoundary found no boundary nodes. This likely indicates an issue "
          "with the binary media buffer or that the geometry is too small for the given radius.");

    // Compute per-node _r from Knudsen number (mantis formula)
    // Use ghosted _local_Knudsen_number so _r shape matches ghosted coords in _boundary_entries
    auto sigma = 1.0 - torch::log10(1.0 + torch::pow(_local_Knudsen_number, 0.7));
    auto sigma_v = (2.0 - sigma) / sigma;
    auto A1 = 1.0 - 0.1817 * sigma_v;
    _r = 1.0 / (1.0 + std::sqrt(libMesh::pi / 6.0) * A1 * sigma_v);
  }

  // Extract index columns
  auto ix = _boundary_entries.index({Slice(), 0});
  auto iy = _boundary_entries.index({Slice(), 1});
  auto iz = _boundary_entries.index({Slice(), 2});
  auto ic = _boundary_entries.index({Slice(), 3});

  // Gather per-entry r values from the [Nx, Ny, Nz] tensor
  auto r_entries = _r.index({ix, iy, iz});

  // Opposite directions for bounce-back
  auto ic_opp = _stencil._op.index_select(0, ic);

  // Gather incoming distributions: f_old[x, y, z, ic]
  auto f_incoming = _f_old[0].index({ix, iy, iz, ic});

  // Bounce-back part: f[x, y, z, opposite(ic)] += r * f_incoming
  _u.index_put_({ix, iy, iz, ic_opp}, r_entries * f_incoming);

  // Specular reflection part: f[x, y, z, sr_dir] += (1 - r) * f_incoming
  _u.index_put_({ix, iy, iz, _specular_directions},
                _u.index({ix, iy, iz, _specular_directions}) + (1.0 - r_entries) * f_incoming);
}

void
LBMSpecularReflectionBoundary::computeBuffer()
{
  const auto n_old = _f_old.size();
  if (n_old == 0)
    return;

  LBMBoundaryCondition::computeBuffer();
  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
