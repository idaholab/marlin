/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "GradientVector.h"

registerMooseObject("MarlinApp", GradientVector);

InputParameters
GradientVector::validParams()
{
  InputParameters params = FFTGradientBase<>::validParams();
  params.addClassDescription(
      "Gradient of the coupled tensor buffer returned as a stacked torch tensor.");
  return params;
}

GradientVector::GradientVector(const InputParameters & parameters) : FFTGradientBase<>(parameters)
{
}

void
GradientVector::computeBuffer()
{
  const auto reciprocal_input = reciprocalInput();
  const auto grad_x = computeGradientComponent(reciprocal_input, 0);
  const auto zero = torch::zeros_like(grad_x);
  _u = torch::stack({grad_x,
                     _dim > 1 ? computeGradientComponent(reciprocal_input, 1) : zero,
                     _dim > 2 ? computeGradientComponent(reciprocal_input, 2) : zero},
                    -1);
}
