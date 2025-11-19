/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperatorBase.h"

/**
 * TensorOperator object with a single output
 */
template <typename T = torch::Tensor>
class TensorOperator : public TensorOperatorBase
{
public:
  static InputParameters validParams();

  TensorOperator(const InputParameters & parameters);

protected:
  /// output buffer
  T & _u;
};

template <typename T>
InputParameters
TensorOperator<T>::validParams()
{
  InputParameters params = TensorOperatorBase::validParams();
  params.addRequiredParam<TensorOutputBufferName>("buffer",
                                                  "The buffer this compute is writing to");
  params.addClassDescription("TensorOperator object.");
  return params;
}

template <typename T>
TensorOperator<T>::TensorOperator(const InputParameters & parameters)
  : TensorOperatorBase(parameters), _u(getOutputBuffer<T>("buffer"))
{
}
