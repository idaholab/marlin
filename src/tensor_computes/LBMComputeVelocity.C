/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMComputeVelocity.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMComputeVelocity);

InputParameters
LBMComputeVelocity::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addRequiredParam<TensorInputBufferName>("f", "Distribution function");
  params.addRequiredParam<TensorInputBufferName>("rho", "Density");
  params.addParam<TensorInputBufferName>("forces", "forces", "Force tensor");
  params.addParam<bool>("enable_forces", false, "Whether to enable forces or no");
  params.addParam<bool>("add_body_force", false, "Whether to enable forces or no");
  params.addParam<MarlinConstantName>("body_force_x", "0.0", "Body force to be added in x-dir");
  params.addParam<MarlinConstantName>("body_force_y", "0.0", "Body force to be added in y-dir");
  params.addParam<MarlinConstantName>("body_force_z", "0.0", "Body force to be added in z-dir");
  params.addClassDescription("Compute object for macroscopic velocity reconstruction.");
  return params;
}

LBMComputeVelocity::LBMComputeVelocity(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _f(getInputBuffer("f", _radius)),
    _rho(getInputBuffer("rho", _radius)),
    _force_tensor(getInputBuffer("forces", _radius)),
    _body_force_constant_x(
        _lb_problem.getConstant<Real>(getParam<MarlinConstantName>("body_force_x"))),
    _body_force_constant_y(
        _lb_problem.getConstant<Real>(getParam<MarlinConstantName>("body_force_y"))),
    _body_force_constant_z(
        _lb_problem.getConstant<Real>(getParam<MarlinConstantName>("body_force_z")))
{
  if (getParam<bool>("add_body_force"))
  {
    std::vector<int64_t> shape = _lb_problem.getLocalTensorShape(std::vector<int64_t>());
    if (shape.size() < 3)
      shape.push_back(1);
    shape.push_back(_dim);

    _body_forces = torch::zeros(shape, MooseTensor::floatTensorOptions());

    auto force_constants =
        torch::tensor({_body_force_constant_x, _body_force_constant_y, _body_force_constant_z},
                      MooseTensor::floatTensorOptions());

    for (int64_t d = 0; d < _dim; d++)
    {
      auto t_index = torch::tensor({d}, MooseTensor::intTensorOptions());
      _body_forces.index_fill_(-1, t_index, force_constants[d]);
    }
  }
}

void
LBMComputeVelocity::computeBuffer()
{
  const unsigned int dim = _domain.getDim();
  const int64_t N = _u.numel() / _u.size(-1);
  const int Q = _f.size(-1);

  auto u_flat = _u.view({N, _u.size(-1)});
  auto f_flat = _f.view({N, Q});
  auto rho_flat = _rho.view({N, 1});

  // u = f @ e_mat
  torch::mm_out(u_flat, f_flat, _e_mat);
  u_flat.div_(rho_flat);

  if (getParam<bool>("enable_forces"))
  {
    auto forces_flat = _force_tensor.slice(-1, 0, dim).reshape({N, dim});
    u_flat.addcdiv_(forces_flat, rho_flat, /*value=*/0.5);
  }
  if (getParam<bool>("add_body_force"))
  {
    auto body_forces_flat = _body_forces.slice(-1, 0, dim).reshape({N, dim});
    u_flat.addcdiv_(body_forces_flat, rho_flat, /*value=*/0.5);
  }

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
