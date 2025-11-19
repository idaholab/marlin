/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SemiImplicitCriticalTimeStep.h"

registerMooseObject("MarlinApp", SemiImplicitCriticalTimeStep);

InputParameters
SemiImplicitCriticalTimeStep::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addClassDescription(
      "Compute the critical timestep given the reciprocal space representation of the linear "
      "operator in a semi-implicit time integrator.");
  params.addParam<Real>("c", 1.0, "Courant number (CFL factor)");
  return params;
}

SemiImplicitCriticalTimeStep::SemiImplicitCriticalTimeStep(const InputParameters & parameters)
  : TensorPostprocessor(parameters)
{
}

void
SemiImplicitCriticalTimeStep::execute()
{
  const auto max_norm_k = std::sqrt(torch::max(_u * _u.conj()).cpu().item<double>());
  _critical_dt = max_norm_k > 0.0 ? 1.0 / max_norm_k : 1e30;
}

void
SemiImplicitCriticalTimeStep::finalize()
{
  gatherMin(_critical_dt);
}

PostprocessorValue
SemiImplicitCriticalTimeStep::getValue() const
{
  return _critical_dt;
}
