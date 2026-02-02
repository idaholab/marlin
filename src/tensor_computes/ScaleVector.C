/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ScaleVector.h"

registerMooseObject("MarlinApp", ScaleVector);

InputParameters
ScaleVector::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Scale a vector tensor by a scalar field.");
  params.addRequiredParam<TensorInputBufferName>("scalar", "Scalar field to multiply with.");
  params.addRequiredParam<TensorInputBufferName>("vector", "Vector field to scale.");
  return params;
}

ScaleVector::ScaleVector(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _scalar(getInputBuffer("scalar")),
    _vector(getInputBuffer("vector"))
{
}

void
ScaleVector::computeBuffer()
{
  if (_vector.dim() < static_cast<int64_t>(_dim) + 1)
    mooseError("ScaleVector vector input must have value_dimensions = '3'.");
  if (_vector.size(_dim) != 3)
    mooseError("ScaleVector vector input must have value_dimensions = '3'.");

  _u = _vector * _scalar.unsqueeze(static_cast<int64_t>(_dim));
}
