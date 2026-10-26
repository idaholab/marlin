/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FiniteDifferenceLaplacian.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", FiniteDifferenceLaplacian);

InputParameters
FiniteDifferenceLaplacian::validParams()
{
  InputParameters params = TensorOperator<torch::Tensor>::validParams();
  params.addRequiredParam<TensorInputBufferName>("input", "The input buffer");
  params.addParam<Real>("factor", 1, "Pre-factor for the Laplacian");
  MooseEnum stencil("3=3 5=5", "3");
  params.addParam<MooseEnum>(
      "stencil_width", stencil, "Stencil width for the Laplacian (3 => 3-point, 5 => 5-point)");
  params.addClassDescription("Computes the Laplacian using finite differences.");
  return params;
}

FiniteDifferenceLaplacian::FiniteDifferenceLaplacian(const InputParameters & parameters)
  : TensorOperator<torch::Tensor>(parameters),
    _radius(getParam<MooseEnum>("stencil_width") == 5 ? 2u : 1u),
    _input_name(getParam<TensorInputBufferName>("input")),
    _u_in(getInputBufferByName(_input_name, _radius)),
    _factor(getParam<Real>("factor"))
{
}

void
FiniteDifferenceLaplacian::realSpaceComputeBuffer()
{
  _tensor_problem.runComputeWithGhosts(*this);
}

void
FiniteDifferenceLaplacian::computeBuffer()
{
  const auto & grid_spacing = _domain.getGridSpacing();

  // Use convolution for efficient stencil application
  // Construct kernel
  torch::Tensor kernel;
  auto options = _u_in.options();

  const double c0_3 = -2.0;
  const double c1_3 = 1.0;

  const double c0_5 = -2.5;      // -5/2
  const double c1_5 = 4.0 / 3.0; // 4/3
  const double c2_5 = -1.0 / 12.0;

  const bool wide = (_radius == 2);

  if (_dim == 1)
  {
    auto dx = grid_spacing(0);
    if (wide)
      kernel = torch::tensor({c2_5, c1_5, c0_5, c1_5, c2_5}, options).view({1, 1, 5}) / (dx * dx);
    else
      kernel = torch::tensor({c1_3, c0_3, c1_3}, options).view({1, 1, 3}) / (dx * dx);

    auto input = _u_in.view({1, 1, _u_in.size(0)});
    auto result = torch::nn::functional::conv1d(
        input, kernel, torch::nn::functional::Conv1dFuncOptions().padding(_radius));
    _u.copy_(result.view(_u.sizes()));
  }
  else if (_dim == 2)
  {
    auto dx = grid_spacing(0);
    auto dy = grid_spacing(1);

    const int ksize = wide ? 5 : 3;
    kernel = torch::zeros({ksize, ksize}, options);
    const int c = ksize / 2;

    auto add_axis_weights = [&](int offset, double coeff, bool xdir)
    {
      if (xdir)
        kernel[c + offset][c] += coeff / (dx * dx);
      else
        kernel[c][c + offset] += coeff / (dy * dy);
    };

    if (wide)
    {
      add_axis_weights(0, c0_5, true);
      add_axis_weights(1, c1_5, true);
      add_axis_weights(-1, c1_5, true);
      add_axis_weights(2, c2_5, true);
      add_axis_weights(-2, c2_5, true);

      add_axis_weights(0, c0_5, false);
      add_axis_weights(1, c1_5, false);
      add_axis_weights(-1, c1_5, false);
      add_axis_weights(2, c2_5, false);
      add_axis_weights(-2, c2_5, false);
    }
    else
    {
      add_axis_weights(0, c0_3, true);
      add_axis_weights(1, c1_3, true);
      add_axis_weights(-1, c1_3, true);

      add_axis_weights(0, c0_3, false);
      add_axis_weights(1, c1_3, false);
      add_axis_weights(-1, c1_3, false);
    }

    kernel = kernel.view({1, 1, ksize, ksize});

    auto input = _u_in.view({1, 1, _u_in.size(0), _u_in.size(1)});
    auto result = torch::nn::functional::conv2d(
        input, kernel, torch::nn::functional::Conv2dFuncOptions().padding(_radius));
    _u.copy_(result.view(_u.sizes()));
  }
  else if (_dim == 3)
  {
    auto dx = grid_spacing(0);
    auto dy = grid_spacing(1);
    auto dz = grid_spacing(2);

    const int ksize = wide ? 5 : 3;
    kernel = torch::zeros({ksize, ksize, ksize}, options);
    const int c = ksize / 2;

    auto add_axis_weights = [&](int offset, double coeff, int axis)
    {
      if (axis == 0)
        kernel[c + offset][c][c] += coeff / (dx * dx);
      else if (axis == 1)
        kernel[c][c + offset][c] += coeff / (dy * dy);
      else
        kernel[c][c][c + offset] += coeff / (dz * dz);
    };

    const auto apply = [&](int axis)
    {
      if (wide)
      {
        add_axis_weights(0, c0_5, axis);
        add_axis_weights(1, c1_5, axis);
        add_axis_weights(-1, c1_5, axis);
        add_axis_weights(2, c2_5, axis);
        add_axis_weights(-2, c2_5, axis);
      }
      else
      {
        add_axis_weights(0, c0_3, axis);
        add_axis_weights(1, c1_3, axis);
        add_axis_weights(-1, c1_3, axis);
      }
    };

    apply(0);
    apply(1);
    apply(2);

    kernel = kernel.view({1, 1, ksize, ksize, ksize});

    auto input = _u_in.view({1, 1, _u_in.size(0), _u_in.size(1), _u_in.size(2)});
    auto result = torch::nn::functional::conv3d(
        input, kernel, torch::nn::functional::Conv3dFuncOptions().padding(_radius));
    _u.copy_(result.view(_u.sizes()));
  }

  if (_factor != 1.0)
    _u *= _factor;
}

std::vector<FDStencilEntry>
FiniteDifferenceLaplacian::stencilAt() const
{
  const auto & grid_spacing = _domain.getGridSpacing();
  std::vector<Real> dx(_dim);
  for (unsigned int axis = 0; axis < _dim; ++axis)
    dx[axis] = grid_spacing(axis);

  return stencilAt(dx);
}

std::vector<FDStencilEntry>
FiniteDifferenceLaplacian::stencilAt(const std::vector<Real> & grid_spacing) const
{
  return buildStencil(_dim, _radius == 2, grid_spacing, _factor);
}

LinearFDOperator::Definiteness
FiniteDifferenceLaplacian::definiteness() const
{
  // The discrete periodic Laplacian (3- or 5-point) is symmetric negative
  // semidefinite with the constants as its nullspace: factor < 0 flips it to
  // positive semidefinite.
  return _factor < 0.0 ? Definiteness::POSITIVE_SEMIDEFINITE : Definiteness::OTHER;
}

std::vector<FDStencilEntry>
FiniteDifferenceLaplacian::buildStencil(unsigned int dim,
                                        bool wide,
                                        const std::vector<Real> & grid_spacing,
                                        Real factor)
{
  // Same coefficients used by computeBuffer() to build its convolution kernel.
  const double c0_3 = -2.0;
  const double c1_3 = 1.0;

  const double c0_5 = -2.5;      // -5/2
  const double c1_5 = 4.0 / 3.0; // 4/3
  const double c2_5 = -1.0 / 12.0;

  std::vector<FDStencilEntry> stencil;

  const auto addAxisEntry = [&](unsigned int axis, int offset, double coeff, double dx)
  {
    std::vector<int> idx(dim, 0);
    idx[axis] = offset;
    stencil.push_back({idx, static_cast<Real>(coeff / (dx * dx) * factor), {}});
  };

  for (unsigned int axis = 0; axis < dim; ++axis)
  {
    const auto dx = grid_spacing[axis];
    if (wide)
    {
      addAxisEntry(axis, 0, c0_5, dx);
      addAxisEntry(axis, 1, c1_5, dx);
      addAxisEntry(axis, -1, c1_5, dx);
      addAxisEntry(axis, 2, c2_5, dx);
      addAxisEntry(axis, -2, c2_5, dx);
    }
    else
    {
      addAxisEntry(axis, 0, c0_3, dx);
      addAxisEntry(axis, 1, c1_3, dx);
      addAxisEntry(axis, -1, c1_3, dx);
    }
  }

  return stencil;
}
