/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMComputePhysicalVelocity.h"
#include "LatticeBoltzmannProblem.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", LBMComputePhysicalVelocity);

InputParameters
LBMComputePhysicalVelocity::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Computes physical velocity based on viscosity and cell size.");
  params.addRequiredParam<TensorInputBufferName>("velocity", "LBM velocity");
  params.addRequiredParam<std::string>("tau", "Relaxation parameter");
  params.addRequiredParam<std::string>("nu", "Physical fluid kinematic viscosity.");
  return params;
}

LBMComputePhysicalVelocity::LBMComputePhysicalVelocity(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _u_lb(getInputBuffer("velocity", _radius)),
    _tau(_tensor_problem.getConstant<Real>(getParam<std::string>("tau"))),
    _nu(_tensor_problem.getConstant<Real>(getParam<std::string>("nu"))),
    _grid_spacing(_domain.getGridSpacing())
{
}

void
LBMComputePhysicalVelocity::computeBuffer()
{
  const Real factor = (1.0 / _lb_problem._cs2 * _nu / (_tau - 0.5));
  const unsigned int & dim = _domain.getDim();
  switch (dim)
  {
    case 2:
      _u = _u_lb * torch::tensor({factor / _grid_spacing(0), factor / _grid_spacing(1)},
                                 MooseTensor::floatTensorOptions());
      break;
    case 3:
    {
      _u = _u_lb *
           torch::tensor(
               {factor / _grid_spacing(0), factor / _grid_spacing(1), factor / _grid_spacing(2)},
               MooseTensor::floatTensorOptions());
      break;
    }
    default:
      mooseError("Unsupported dimension");
  }

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
