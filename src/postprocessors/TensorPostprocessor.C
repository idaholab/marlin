/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorPostprocessor.h"
#include "TensorProblem.h"

template <class T>
InputParameters
TensorPostprocessorTempl<T>::validParams()
{
  InputParameters params = T::validParams();
  params.addClassDescription("A normal Postprocessor acting on a Tensor buffer.");
  params.addRequiredParam<TensorInputBufferName>("buffer", "The buffer this compute is operating on");
  return params;
}

template <class T>
TensorPostprocessorTempl<T>::TensorPostprocessorTempl(const InputParameters & parameters)
  : T(parameters),
    DomainInterface(this),
    _tensor_problem(TensorProblem::cast(this, this->_fe_problem)),
    _buffer_base(
        _tensor_problem.getBufferBase(this->template getParam<TensorInputBufferName>("buffer"))),
    _u(_buffer_base.getRawTensor())
{
}

template class TensorPostprocessorTempl<GeneralPostprocessor>;
template class TensorPostprocessorTempl<GeneralVectorPostprocessor>;
