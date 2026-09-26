/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "PenaltyRigidWallNodalKernel.h"

registerMooseObject("MarlinApp", PenaltyRigidWallNodalKernel);

InputParameters
PenaltyRigidWallNodalKernel::validParams()
{
  InputParameters params = NodalKernel::validParams();
  params.addClassDescription(
      "One-sided rigid-wall penalty contact applied at every node: a node whose current "
      "position along the given component drops below the wall plane receives a restoring "
      "force F = -penalty * penetration. Holds interior nodes as well as surface nodes.");
  params.addRequiredRangeCheckedParam<Real>(
      "penalty", "penalty > 0", "Penalty stiffness per node (N/m).");
  params.addParam<Real>("wall_position", 0.0, "Wall plane position along the component.");
  params.addRequiredRangeCheckedParam<unsigned int>(
      "component",
      "component <= 2",
      "Coordinate component the variable displaces (0=x, 1=y, 2=z). Must match the "
      "displacement variable this kernel is applied to.");
  return params;
}

PenaltyRigidWallNodalKernel::PenaltyRigidWallNodalKernel(const InputParameters & parameters)
  : NodalKernel(parameters),
    _penalty(getParam<Real>("penalty")),
    _wall_position(getParam<Real>("wall_position")),
    _component(getParam<unsigned int>("component"))
{
}

Real
PenaltyRigidWallNodalKernel::penetration() const
{
  // current position = reference position + displacement; negative when below the wall
  return (*_current_node)(_component) + _u[_qp] - _wall_position;
}

Real
PenaltyRigidWallNodalKernel::computeQpResidual()
{
  const auto pen = penetration();
  // restoring force F = -penalty * pen (points back toward the wall); residual = -F
  return pen < 0 ? _penalty * pen : 0.0;
}

Real
PenaltyRigidWallNodalKernel::computeQpJacobian()
{
  return penetration() < 0 ? _penalty : 0.0;
}
