/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FiniteDifferenceDivGrad.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", FiniteDifferenceDivGrad);

InputParameters
FiniteDifferenceDivGrad::validParams()
{
  InputParameters params = TensorOperator<torch::Tensor>::validParams();
  params.addRequiredParam<TensorInputBufferName>("input", "The input buffer u");
  params.addRequiredParam<TensorInputBufferName>("coefficient",
                                                 "The cell-centred coefficient buffer D");
  params.addParam<Real>("factor", 1, "Pre-factor for div(D grad u)");
  MooseEnum face_mean("harmonic arithmetic", "harmonic");
  params.addParam<MooseEnum>(
      "face_mean",
      face_mean,
      "Mean of the two cells' D on a face: harmonic (exact for a jump in D on a cell face, "
      "requires D_i D_j > 0) or arithmetic (for a D that can reach zero).");
  params.addClassDescription(
      "Computes div(D grad u) for a cell-centred coefficient D using finite differences.");
  return params;
}

FiniteDifferenceDivGrad::FiniteDifferenceDivGrad(const InputParameters & parameters)
  : TensorOperator<torch::Tensor>(parameters),
    _input_name(getParam<TensorInputBufferName>("input")),
    _u_in(getInputBufferByName(_input_name, 1)),
    _coefficient_name(getParam<TensorInputBufferName>("coefficient")),
    // D's halo is filled by exchangeCoefficient, not from boundary conditions
    _coefficient(getInputBufferByName(_coefficient_name, 0)),
    _factor(getParam<Real>("factor")),
    _harmonic(getParam<MooseEnum>("face_mean") == "harmonic")
{
  if (!_domain.isRealSpaceMode())
    mooseError("FiniteDifferenceDivGrad requires parallel_mode = REAL_SPACE.");
}

void
FiniteDifferenceDivGrad::realSpaceComputeBuffer()
{
  _tensor_problem.runComputeWithGhosts(*this);
}

torch::Tensor
FiniteDifferenceDivGrad::ownedBlock(const torch::Tensor & t, unsigned int axis, int64_t shift) const
{
  const auto halo = int64_t(_tensor_problem.getMaxGhostLayer());
  const auto & owned = _domain.getLocalGridSize();
  auto block = t;
  for (const auto d : make_range(_dim))
    block = block.narrow(d, halo + (d == axis ? shift : 0), owned[d]);
  return block;
}

void
FiniteDifferenceDivGrad::exchangeCoefficient() const
{
  // As face data: the halo along every axis (_dim is no axis) from the neighbouring ranks and
  // periodic images, and the nearest owned value on a domain face, which is D_0 there.
  _tensor_problem.exchangeGhostLayers(_coefficient_name, 1, int(_dim));
}

std::pair<torch::Tensor, torch::Tensor>
FiniteDifferenceDivGrad::faceCoefficients(unsigned int axis) const
{
  const auto D0 = ownedBlock(_coefficient);
  const auto mean = [this, &D0](const torch::Tensor & Dj)
  { return _harmonic ? 2.0 * D0 * Dj / (D0 + Dj) : 0.5 * (D0 + Dj); };
  return {mean(ownedBlock(_coefficient, axis, -1)), mean(ownedBlock(_coefficient, axis, 1))};
}

void
FiniteDifferenceDivGrad::computeBuffer()
{
  exchangeCoefficient();
  const auto & grid_spacing = _domain.getGridSpacing();
  const auto u0 = ownedBlock(_u_in);

  // the harmonic mean needs D_i D_j > 0 on every face an owned cell reads
  if (_harmonic)
    for (const auto d : make_range(_dim))
    {
      const auto D0 = ownedBlock(_coefficient);
      if (torch::any(D0 * ownedBlock(_coefficient, d, -1) <= 0).item<bool>() ||
          torch::any(D0 * ownedBlock(_coefficient, d, 1) <= 0).item<bool>())
        mooseError("The coefficient '",
                   _coefficient_name,
                   "' of FiniteDifferenceDivGrad '",
                   name(),
                   "' has D_i D_j <= 0 on a face, where the harmonic mean is undefined. Use "
                   "face_mean = arithmetic for a coefficient that changes sign or reaches zero.");
    }

  auto result = torch::zeros_like(u0);
  for (const auto d : make_range(_dim))
  {
    const auto [D_lower, D_upper] = faceCoefficients(d);
    const auto h = grid_spacing(d);
    result +=
        (D_upper * (ownedBlock(_u_in, d, 1) - u0) - D_lower * (u0 - ownedBlock(_u_in, d, -1))) /
        (h * h);
  }
  ownedBlock(_u).copy_(_factor * result);
}

std::vector<FDStencilEntry>
FiniteDifferenceDivGrad::stencilAt() const
{
  const auto & grid_spacing = _domain.getGridSpacing();
  std::vector<Real> dx(_dim);
  for (const auto d : make_range(_dim))
    dx[d] = grid_spacing(d);
  return stencilAt(dx);
}

std::vector<FDStencilEntry>
FiniteDifferenceDivGrad::stencilAt(const std::vector<Real> & grid_spacing) const
{
  // the fields are on this rank's owned block of the fine grid
  const auto & fine_spacing = _domain.getGridSpacing();
  for (const auto d : make_range(_dim))
    if (grid_spacing[d] != fine_spacing(d))
      mooseError("FiniteDifferenceDivGrad '", name(), "' has no coarse-level stencil.");

  // an implicit solve needs a semidefinite operator: D > 0 (D >= 0 with the arithmetic mean)
  const auto D0 = ownedBlock(_coefficient);
  if (_harmonic ? torch::any(D0 <= 0).item<bool>() : torch::any(D0 < 0).item<bool>())
    mooseError("The coefficient '",
               _coefficient_name,
               "' of FiniteDifferenceDivGrad '",
               name(),
               _harmonic ? "' must be positive in an implicit solve with the harmonic face mean."
                         : "' must be nonnegative in an implicit solve.");

  exchangeCoefficient();
  std::vector<FDStencilEntry> stencil;
  auto centre = torch::zeros_like(D0);
  for (const auto d : make_range(_dim))
  {
    const auto [D_lower, D_upper] = faceCoefficients(d);
    const auto h2 = grid_spacing[d] * grid_spacing[d];
    std::vector<int> lower(_dim, 0), upper(_dim, 0);
    lower[d] = -1;
    upper[d] = 1;
    stencil.push_back({lower, _factor / h2, D_lower.contiguous()});
    stencil.push_back({upper, _factor / h2, D_upper.contiguous()});
    centre -= (D_lower + D_upper) / h2;
  }
  stencil.push_back({std::vector<int>(_dim, 0), _factor, centre});
  return stencil;
}

std::vector<std::vector<int>>
FiniteDifferenceDivGrad::stencilOffsets() const
{
  std::vector<std::vector<int>> offsets{std::vector<int>(_dim, 0)};
  for (const auto d : make_range(_dim))
    for (const int o : {-1, 1})
    {
      std::vector<int> offset(_dim, 0);
      offset[d] = o;
      offsets.push_back(offset);
    }
  return offsets;
}

LinearFDOperator::Definiteness
FiniteDifferenceDivGrad::definiteness() const
{
  // with D >= 0 (checked by stencilAt) div(D grad u) is symmetric negative semidefinite on a
  // periodic grid, the constants in its nullspace
  return _factor < 0.0 ? Definiteness::POSITIVE_SEMIDEFINITE : Definiteness::OTHER;
}
