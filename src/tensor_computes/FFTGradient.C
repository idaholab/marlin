/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FFTGradient.h"

registerMooseObject("MarlinApp", FFTGradient);

InputParameters
FFTGradient::validParams()
{
  InputParameters params = FFTGradientBase<>::validParams();
  params.addClassDescription("Tensor gradient.");
  params.addRequiredParam<MooseEnum>(
      "direction", MooseEnum("X=0 Y=1 Z=2"), "Which axis to take the gradient along.");
  return params;
}

FFTGradient::FFTGradient(const InputParameters & parameters)
  : FFTGradientBase<>(parameters), _direction(getParam<MooseEnum>("direction"))
{
}

void
FFTGradient::computeBuffer()
{
  _u = computeGradientComponent(_direction);
}
