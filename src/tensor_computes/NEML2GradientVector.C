/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MooseError.h"
#include "NEML2GradientVector.h"

registerMooseObject("MarlinApp", NEML2GradientVector);

InputParameters
NEML2GradientVector::validParams()
{
  InputParameters params = FFTGradientBase<NEML2GradientVectorType>::validParams();
#ifdef NEML2_ENABLED
  params.addClassDescription("Gradient of the coupled tensor buffer returned as a NEML2 vector.");
#else
  params.addClassDescription("Object requires NEML2.");
#endif
  return params;
}

NEML2GradientVector::NEML2GradientVector(const InputParameters & parameters)
  : FFTGradientBase<NEML2GradientVectorType>(parameters),
    _zero(torch::tensor(0.0, MooseTensor::floatTensorOptions()))
{
#ifndef NEML2_ENABLED
  mooseError("Object requires NEML2");
#endif
}

void
NEML2GradientVector::computeBuffer()
{
#ifdef NEML2_ENABLED
  const auto reciprocal_input = reciprocalInput();
  auto grad_x = computeGradientComponent(reciprocal_input, 0);
  auto grad_y = (_dim > 1) ? computeGradientComponent(reciprocal_input, 1) : torch::zeros_like(grad_x);
  auto grad_z = (_dim > 2) ? computeGradientComponent(reciprocal_input, 2) : torch::zeros_like(grad_x);
  _u = torch::stack({grad_x, grad_y, grad_z}, -1).contiguous();
#endif
}
