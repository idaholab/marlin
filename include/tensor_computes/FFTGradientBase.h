/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * Shared FFT gradient implementation.
 *
 * This base class is intentionally not registered as a MOOSE object.
 */
template <typename T = torch::Tensor>
class FFTGradientBase : public TensorOperator<T>
{
public:
  static InputParameters validParams();

  FFTGradientBase(const InputParameters & parameters);

  /// Parallel FFT uses MPI communication which cannot be JIT traced
  virtual bool supportsJIT() const override { return !this->usesParallelFFT(); }

protected:
  torch::Tensor computeGradientComponent(unsigned int direction) const;
  torch::Tensor computeGradientComponent(const torch::Tensor & reciprocal_input,
                                         unsigned int direction) const;
  torch::Tensor reciprocalInput() const;

  using TensorOperator<T>::_u;
  using TensorOperator<T>::_domain;
  using TensorOperator<T>::getInputBuffer;

  const torch::Tensor & _input;
  const bool _input_is_reciprocal;

  /// imaginary unit i
  const torch::Tensor _imaginary_unit;
};

template <typename T>
InputParameters
FFTGradientBase<T>::validParams()
{
  InputParameters params = TensorOperator<T>::validParams();
  params.addRequiredParam<TensorInputBufferName>("input", "Input buffer name");
  params.addParam<bool>(
      "input_is_reciprocal", false, "Input buffer is already in reciprocal space");
  return params;
}

template <typename T>
FFTGradientBase<T>::FFTGradientBase(const InputParameters & parameters)
  : TensorOperator<T>(parameters),
    _input(getInputBuffer("input")),
    _input_is_reciprocal(this->template getParam<bool>("input_is_reciprocal")),
    _imaginary_unit(
        torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions()))
{
}

template <typename T>
torch::Tensor
FFTGradientBase<T>::reciprocalInput() const
{
  return _input_is_reciprocal ? _input : _domain.fft(_input);
}

template <typename T>
torch::Tensor
FFTGradientBase<T>::computeGradientComponent(unsigned int direction) const
{
  return computeGradientComponent(reciprocalInput(), direction);
}

template <typename T>
torch::Tensor
FFTGradientBase<T>::computeGradientComponent(const torch::Tensor & reciprocal_input,
                                             unsigned int direction) const
{
  return _domain.ifft(reciprocal_input * _domain.getReciprocalAxis(direction) * _imaginary_unit);
}
