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
        _lb_problem.getConstant<Real>(getParam<MarlinConstantName>("body_force_z"))),
    _enable_forces(getParam<bool>("enable_forces")),
    _add_body_force(getParam<bool>("add_body_force"))
{
  // constant body force [1, dim], broadcast over all nodes
  if (_add_body_force)
    _body_forces =
        torch::tensor({_body_force_constant_x, _body_force_constant_y, _body_force_constant_z},
                      MooseTensor::floatTensorOptions())
            .slice(0, 0, _dim)
            .view({1, _dim});
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

  if (_enable_forces)
  {
    auto forces_flat = _force_tensor.slice(-1, 0, dim).reshape({N, dim});
    u_flat.addcdiv_(forces_flat, rho_flat, /*value=*/0.5);
  }
  if (_add_body_force)
  {
    u_flat.addcdiv_(_body_forces, rho_flat, /*value=*/0.5);
  }

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, _solid_values);
}
