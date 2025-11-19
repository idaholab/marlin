/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ComputeReynoldsNumber.h"
#include "TensorProblem.h"

registerMooseObject("MarlinApp", ComputeReynoldsNumber);

InputParameters
ComputeReynoldsNumber::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addRequiredParam<std::string>("tau", "Fluid viscosity");
  params.addRequiredParam<std::string>("diameter", "Characteristic diamaeter");
  params.addClassDescription("Compute Reynolds number.");
  return params;
}

ComputeReynoldsNumber::ComputeReynoldsNumber(const InputParameters & parameters)
  : TensorPostprocessor(parameters),
    _tau(_tensor_problem.getConstant<Real>(getParam<std::string>("tau"))),
    _D(_tensor_problem.getConstant<Real>(getParam<std::string>("diameter")))
{
}

void
ComputeReynoldsNumber::execute()
{
  const Real kinematic_viscosity = 1.0 / sqrt(3.0) * (_tau - 0.5);
  const auto avg_speed = _u.sum().cpu().item<double>() / torch::numel(_u);
  _Reynolds_number = avg_speed * _D / kinematic_viscosity;
}

PostprocessorValue
ComputeReynoldsNumber::getValue() const
{
  return _Reynolds_number;
}
