/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FiniteDifferenceBiharmonic.h"
#include "FiniteDifferenceLaplacian.h"
#include "TensorProblem.h"
#include "TensorRobinBC.h"
#include "DomainAction.h"

#include <map>

registerMooseObject("MarlinApp", FiniteDifferenceBiharmonic);

InputParameters
FiniteDifferenceBiharmonic::validParams()
{
  InputParameters params = TensorOperator<torch::Tensor>::validParams();
  params.addRequiredParam<TensorInputBufferName>("input", "The input buffer");
  params.addParam<Real>("factor", 1, "Pre-factor for the biharmonic");
  MooseEnum stencil("3=3 5=5", "3");
  params.addParam<MooseEnum>("stencil_width",
                             stencil,
                             "Stencil width of the Laplacian that is applied twice (3 => 3-point, "
                             "5 => 5-point)");
  params.addClassDescription(
      "Computes the biharmonic Laplacian(Laplacian(u)) using finite differences.");
  return params;
}

FiniteDifferenceBiharmonic::FiniteDifferenceBiharmonic(const InputParameters & parameters)
  : TensorOperator<torch::Tensor>(parameters),
    _laplacian_radius(getParam<MooseEnum>("stencil_width") == 5 ? 2u : 1u),
    _input_name(getParam<TensorInputBufferName>("input")),
    _u_in(getInputBufferByName(_input_name, 2 * _laplacian_radius)),
    _factor(getParam<Real>("factor")),
    _two_stage(false)
{
  // convolution kernel from the stencil, offset o at kernel index o + radius; the grid
  // spacing is fixed once the domain is set up, so it is built here, on the device
  const int64_t radius = 2 * _laplacian_radius;
  const int64_t ksize = 2 * radius + 1;
  std::vector<int64_t> kernel_shape{1, 1};
  kernel_shape.insert(kernel_shape.end(), _dim, ksize);
  auto kernel = torch::zeros(kernel_shape, torch::TensorOptions().dtype(torch::kFloat64));
  auto flat = kernel.view({-1});
  for (const auto & entry : stencilAt())
  {
    int64_t index = 0;
    for (const auto d : make_range(_dim))
      index = index * ksize + entry.offset[d] + radius;
    flat[index] = entry.coefficient;
  }
  _kernel = kernel.to(MooseTensor::floatTensorOptions());
}

void
FiniteDifferenceBiharmonic::init()
{
  // The ghost layers of u from a Dirichlet or Neumann condition give u = c or du/dn = c on the
  // face, but the 13-point stencil applied to them makes the intermediate Laplacian w's ghost
  // the reflection of w only for face data that is linear along the face (the pair imposed is
  // d2u/dn2 = 0 or d3u/dn3 = 0). Setting w's ghosts to -w0 (Dirichlet) or w0 (Neumann) imposes
  // Laplacian(u) = 0 or dLaplacian(u)/dn = 0 for any face data. With zero face data both give
  // the same operator, so only the boundary-data term changes.
  _w_reflection.assign(2 * _dim, 0);
  if (_laplacian_radius != 1 || !_domain.isRealSpaceMode())
    return;
  std::array<int64_t, 3> begin, end;
  _domain.getLocalBounds(_domain.comm().rank(), begin, end);
  const auto & n = _domain.getGridSize();
  const auto & periodic = _domain.getPeriodicDirections();
  for (const auto face : make_range(2 * _dim))
  {
    const auto d = face / 2;
    const auto * bc = dynamic_cast<const TensorRobinBC *>(
        _tensor_problem.getBoundaryCondition(_input_name, face));
    if (periodic[d] || !bc || (bc->a() != 0.0 && bc->b() != 0.0))
      continue;
    // the face is a domain face of every rank's problem; the reflection is on its ranks only
    _two_stage = true;
    if (face % 2 == 0 ? begin[d] == 0 : end[d] == n[d])
      _w_reflection[face] = bc->b() == 0.0 ? -1 : 1;
  }
}

void
FiniteDifferenceBiharmonic::realSpaceComputeBuffer()
{
  _tensor_problem.runComputeWithGhosts(*this);
}

void
FiniteDifferenceBiharmonic::computeBuffer()
{
  if (_two_stage)
  {
    const auto & dx = _domain.getGridSpacing();
    // the 3-point Laplacian of t on its interior (one cell less on each side of every axis)
    const auto laplacian = [this, &dx](const torch::Tensor & t)
    {
      const auto interior = [this, &t](unsigned int axis, int64_t shift)
      {
        auto b = t;
        for (const auto d : make_range(_dim))
          b = b.narrow(d, 1 + (d == axis ? shift : 0), t.size(d) - 2);
        return b;
      };
      const auto centre = interior(0, 0);
      auto result = torch::zeros_like(centre);
      for (const auto d : make_range(_dim))
        result += (interior(d, 1) - 2.0 * centre + interior(d, -1)) / (dx(d) * dx(d));
      return result;
    };

    // w on the owned cells and one layer around them, from u and its two ghost layers
    const auto halo = int64_t(_tensor_problem.getMaxGhostLayer());
    const auto & owned = _domain.getLocalGridSize();
    auto u = _u_in;
    for (const auto d : make_range(_dim))
      u = u.narrow(d, halo - 2, owned[d] + 4);
    auto w = laplacian(u);
    for (const auto face : make_range(2 * _dim))
      if (_w_reflection[face] != 0)
      {
        const auto d = face / 2;
        const bool upper = face % 2 == 1;
        w.narrow(d, upper ? owned[d] + 1 : 0, 1)
            .copy_(_w_reflection[face] * w.narrow(d, upper ? owned[d] : 1, 1));
      }

    auto out = _u;
    for (const auto d : make_range(_dim))
      out = out.narrow(d, halo, owned[d]);
    out.copy_(_factor * laplacian(w));
    return;
  }

  const int64_t radius = 2 * _laplacian_radius;
  std::vector<int64_t> input_view{1, 1};
  input_view.insert(input_view.end(), _u_in.sizes().begin(), _u_in.sizes().end());
  // conv1d/2d/3d by dimension of the kernel
  const auto result = at::convolution(_u_in.view(input_view),
                                      _kernel,
                                      {},
                                      std::vector<int64_t>(_dim, 1),
                                      std::vector<int64_t>(_dim, radius),
                                      std::vector<int64_t>(_dim, 1),
                                      false,
                                      std::vector<int64_t>(_dim, 0),
                                      1);
  _u.copy_(result.view(_u.sizes()));
}

std::vector<FDStencilEntry>
FiniteDifferenceBiharmonic::stencilAt() const
{
  const auto & grid_spacing = _domain.getGridSpacing();
  std::vector<Real> dx(_dim);
  for (const auto d : make_range(_dim))
    dx[d] = grid_spacing(d);
  return stencilAt(dx);
}

std::vector<FDStencilEntry>
FiniteDifferenceBiharmonic::stencilAt(const std::vector<Real> & grid_spacing) const
{
  return buildStencil(_dim, _laplacian_radius == 2, grid_spacing, _factor);
}

LinearFDOperator::Definiteness
FiniteDifferenceBiharmonic::definiteness() const
{
  // the square of the symmetric periodic Laplacian is positive semidefinite with
  // the constants as its nullspace
  return _factor > 0.0 ? Definiteness::POSITIVE_SEMIDEFINITE : Definiteness::OTHER;
}

std::vector<FDStencilEntry>
FiniteDifferenceBiharmonic::buildStencil(unsigned int dim,
                                         bool wide,
                                         const std::vector<Real> & grid_spacing,
                                         Real factor)
{
  const auto laplacian = FiniteDifferenceLaplacian::buildStencil(dim, wide, grid_spacing, 1.0);

  std::map<std::vector<int>, Real> composed;
  for (const auto & a : laplacian)
    for (const auto & b : laplacian)
    {
      std::vector<int> offset(dim);
      for (const auto d : make_range(dim))
        offset[d] = a.offset[d] + b.offset[d];
      composed[offset] += a.coefficient * b.coefficient;
    }

  std::vector<FDStencilEntry> stencil;
  for (const auto & [offset, coefficient] : composed)
    stencil.push_back({offset, factor * coefficient, {}});
  return stencil;
}
