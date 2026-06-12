/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LatentHeatSource.h"
#include "MarlinUtils.h"

registerMooseObject("MarlinApp", LatentHeatSource);

InputParameters
LatentHeatSource::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Latent heat source L*(s - s_old)/dt, where s_old is the solid "
                             "fraction at the previous solver substep.");
  params.addRequiredParam<TensorInputBufferName>("s", "Solid fraction buffer.");
  params.addParam<Real>("L", 1.0, "Latent heat coefficient.");
  return params;
}

LatentHeatSource::LatentHeatSource(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _s(getInputBuffer("s")),
    _s_old(_tensor_problem.getBufferOld(getParam<TensorInputBufferName>("s"), 1)),
    _L(getParam<Real>("L"))
{
}

void
LatentHeatSource::computeBuffer()
{
  // old buffer states advance once per solver substep, so the rate must use the substep dt
  const auto dt = _tensor_problem.subDt();
  if (_s_old.empty() || dt == 0.0)
  {
    _u = torch::zeros_like(_s);
    return;
  }

  _u = _L * (_s - _s_old[0]) / dt;
}
