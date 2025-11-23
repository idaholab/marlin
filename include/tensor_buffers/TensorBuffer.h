/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
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

  T & getTensor(unsigned int ghost_layers = 0)
  {
    // We always return the full padded tensor.
    // The user is responsible for handling the padding if they requested fewer layers.
    // But typically they request max or 0, and if 0, they should use getUnpaddedTensor()
    // if they want to be strict, or just operate on the padded one.
    // Actually, ParsedCompute requests 0 but operates on whatever it gets.
    // FiniteDifferenceLaplacian requests >0 and expects padding.
    return _u;
  }
  const std::vector<T> & getOldTensor(std::size_t states_requested);

  virtual const torch::Tensor & getRawTensor() const override;
  virtual const torch::Tensor & getRawCPUTensor() override;

  void requestGhostLayers(unsigned int layers)
  {
    _max_ghost_layers = std::max(_max_ghost_layers, layers);
  }
  unsigned int getMaxGhostLayers() const { return _max_ghost_layers; }

  // Return by value (view)
  virtual torch::Tensor getUnpaddedTensor() override
  {
    if (_unpadded_slice.empty())
      return _u;
    return _u.index(_unpadded_slice);
  }

  virtual torch::Tensor getUnpaddedCPUTensor() override
  {
    if (_unpadded_slice.empty())
      return _u_cpu;
    return _u_cpu.index(_unpadded_slice);
  }

  virtual void requestCPUCopy() override { _cpu_copy_requested = true; }

protected:
  /// current state of the tensor (interior + padding)
  T _u;

  /// slice indices for the unpadded view
  std::vector<torch::indexing::TensorIndex> _unpadded_slice;

  /// max ghost layers requested
  unsigned int _max_ghost_layers = 0;


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
