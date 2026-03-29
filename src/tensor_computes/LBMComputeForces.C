/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMComputeForces.h"
#include "LatticeBoltzmannProblem.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMComputeForces);

InputParameters
LBMComputeForces::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  // params.addRequiredParam<TensorInputBufferName>("f", "Distribution function");
  params.addParam<TensorInputBufferName>("temperature", "T", "Macroscopic temperature");
  params.addParam<TensorInputBufferName>("rho", "rho", "Macroscopic density");

  params.addParam<std::string>("rho0", "1.0", "Reference density");
  params.addParam<std::string>("T0", "1.0", "Reference temperature");
  params.addParam<std::string>("gravity", "0.001", "Gravitational accelaration");
  params.addParam<Real>("gravity_direction", 1, "Gravitational accelaration direction");

  params.addParam<bool>("enable_gravity", false, "Whether to consider gravity");
  params.addParam<bool>("enable_buoyancy", false, "Whether to consider buoyancy");

  params.addClassDescription("Compute object for LB forces");
  return params;
}

LBMComputeForces::LBMComputeForces(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _reference_density((_lb_problem.getConstant<Real>(getParam<std::string>("rho0")))),
    _reference_temperature((_lb_problem.getConstant<Real>(getParam<std::string>("T0")))),
    _enable_gravity(getParam<bool>("enable_gravity")),
    _enable_buoyancy(getParam<bool>("enable_buoyancy")),
    _g(_lb_problem.getConstant<Real>(getParam<std::string>("gravity"))),
    _gravity_direction(static_cast<int64_t>(getParam<Real>("gravity_direction"))),
    _density_tensor(getInputBufferByName(getParam<TensorInputBufferName>("rho"), _radius)),
    _temperature(getInputBufferByName(getParam<TensorInputBufferName>("temperature"), _radius))
{
  _buoyancy_const = _g * _reference_density;
  _buoyancy_offset = _buoyancy_const * _reference_temperature;
}

void
LBMComputeForces::computeGravity()
{
  const int64_t N = _u.numel() / _u.size(-1);
  auto u_dir_flat = _u.select(-1, _gravity_direction).view({N});
  auto rho_flat = _density_tensor.view({N});

  u_dir_flat.add_(rho_flat, /*alpha=*/_g);
}

void
LBMComputeForces::computeBuoyancy()
{
  const int64_t N = _u.numel() / _u.size(-1);
  auto u_dir_flat = _u.select(-1, _gravity_direction).view({N});
  auto temp_flat = _temperature.view({N});

  u_dir_flat.add_(temp_flat, /*alpha=*/_buoyancy_const);
  u_dir_flat.sub_(_buoyancy_offset);
}

void
LBMComputeForces::computeBuffer()
{
  _u.zero_();

  if (_enable_gravity)
    computeGravity();
  if (_enable_buoyancy)
    computeBuoyancy();

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
