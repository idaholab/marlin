/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorBufferBase.h"

/**
 * Tensor wrapper arbitrary tensor value dimensions
 */
template <typename T>
class TensorBuffer : public TensorBufferBase
{
public:
  static InputParameters validParams();

  TensorBuffer(const InputParameters & parameters);

  virtual std::size_t advanceState() override;
  virtual void clearStates() override;

  T & getTensor();
  const std::vector<T> & getOldTensor(std::size_t states_requested);

  virtual const torch::Tensor & getRawTensor() const override;
  virtual const torch::Tensor & getRawCPUTensor() override;

protected:
  /// current state of the tensor
  T _u;

  /// potential CPU copy of the tensor (if requested)
  T _u_cpu;

  /// was a CPU copy requested?
  bool _cpu_copy_requested;

  /// old states of the tensor
  std::vector<T> _u_old;
  std::size_t _max_states;
};

template <typename T>
InputParameters
TensorBuffer<T>::validParams()
{
  InputParameters params = TensorBufferBase::validParams();
  return params;
}

template <typename T>
TensorBuffer<T>::TensorBuffer(const InputParameters & parameters)
  : TensorBufferBase(parameters), _cpu_copy_requested(false), _max_states(0)
{
}

template <typename T>
std::size_t
TensorBuffer<T>::advanceState()
{
  // make room to push state one step further back
  if (_u_old.size() < _max_states)
    _u_old.resize(_u_old.size() + 1);

  // push state further back
  if (!_u_old.empty())
  {
    for (std::size_t i = _u_old.size() - 1; i > 0; --i)
      _u_old[i] = _u_old[i - 1];
    _u_old[0] = _u;
  }

  return _u_old.size();
}

template <typename T>
void
TensorBuffer<T>::clearStates()
{
  _u_old.clear();
}

template <typename T>
const torch::Tensor &
TensorBuffer<T>::getRawTensor() const
{
  return _u;
}

template <typename T>
const torch::Tensor &
TensorBuffer<T>::getRawCPUTensor()
{
  _cpu_copy_requested = true;
  return _u_cpu;
}

template <typename T>
T &
TensorBuffer<T>::getTensor()
{
  return _u;
}

template <typename T>
const std::vector<T> &
TensorBuffer<T>::getOldTensor(std::size_t states_requested)
{
  _max_states = std::max(_max_states, states_requested);
  return _u_old;
}

/**
 * Specialization of this helper struct can be used to force the use of derived
 * classes for implicit TensorBuffer construction (i.e. tensors that are not explicitly
 * listed under [TensorBuffers]).
 */
template <typename T>
struct TensorBufferSpecialization;

#define registerTensorType(derived_class, tensor_type)                                             \
  template <>                                                                                      \
  struct TensorBufferSpecialization<tensor_type>                                                   \
  {                                                                                                \
    using type = derived_class;                                                                    \
  }
