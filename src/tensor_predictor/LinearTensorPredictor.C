/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LinearTensorPredictor.h"

InputParameters
LinearTensorPredictor::validParams()
{
  InputParameters params = TensorPredictor::validParams();
  params.addParam<Real>("scale", 1.0, "The scale factor for the predictor (can range from 0 to 1)");
  params.set<unsigned int>("history_size") = 2;
  return params;
}

LinearTensorPredictor::LinearTensorPredictor(const InputParameters & parameters)
  : TensorPredictor(parameters), _scale(getParam<Real>("scale"))
{
}

void
LinearTensorPredictor::computeBuffer()
{
  if (_u_old.size() > 1)
  {
    // compute difference between the old and older steps
    const auto diff = _u_old[0] - _u_old[1];
    if (_scale == 1.0)
      _u = _u + diff;
    else
      _u = _u + diff * _scale;
  }
}
