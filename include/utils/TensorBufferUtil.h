/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "NEML2Utils.h"
#include "MarlinUtils.h"
// #include "neml2/tensors/Scalar.h"

#if 0
using neml2::TensorShapeRef;

namespace MooseTensor
{

template <typename T, std::size_t D>
class TensorBuffer;

template <typename T = torch::Tensor, std::size_t D>
TensorBuffer<T, D>
createBuffer(long int const (&n)[D], Real const (&min)[D], Real const (&max)[D])
{
  TensorShapeRef nn(n);
  std::array<Real, D> amin, amax;
  for (std::size_t i = 0; i < D; ++i)
  {
    amin[i] = min[i];
    amax[i] = max[i];
  }

  return TensorBuffer<T, D>(nn, amin, amax);
}

template <typename T = torch::Tensor, std::size_t D>
TensorBuffer<T, D>
createBuffer(long int const (&n)[D], Real const (&max)[D])
{
  Real min[D];
  for (std::size_t i = 0; i < D; ++i)
    min[i] = 0.0;
  return createBuffer(n, min, max);
}

template <typename T = torch::Tensor, std::size_t D>
TensorBuffer<T, D>
createBuffer(long int const (&n)[D])
{
  Real max[D];
  for (std::size_t i = 0; i < D; ++i)
    max[i] = 1.0;
  return createBuffer(n, max);
}

enum class Interval
{
  CLOSED,
  OPEN,
  LEFT_OPEN,
  RIGHT_OPEN
};

template <typename T, std::size_t D>
class TensorBuffer
{
  TensorBuffer(const TensorShapeRef & batch_shape,
            const std::array<Real, D> & min,
            const std::array<Real, D> & max);

public:
  T & data() { return _data; }
  const T & data() const { return _data; }

  std::array<neml2::Real, D> & min() { return _min; }
  std::array<neml2::Real, D> & max() { return _max; }

  torch::Tensor getAxis(std::size_t dim, Interval interval = Interval::LEFT_OPEN) const;
  torch::Tensor getFrequency(std::size_t dim) const;
  torch::Tensor grad(std::size_t dim, T * fft = nullptr) const;

  auto getAxis() const { return getAxisHelper(std::make_integer_sequence<std::size_t, D>{}); }
  auto getFrequency() const
  {
    return getFrequencyHelper(std::make_integer_sequence<std::size_t, D>{});
  }
  auto grad() const { return gradHelper(std::make_integer_sequence<std::size_t, D>{}); }

  torch::Tensor laplace() const;

  // stream in gnuplot readable format

  friend TensorBuffer<T, D>
  createBuffer<T, D>(long int const (&n)[D], Real const (&min)[D], Real const (&max)[D]);

protected:
  T fft() const;
  T ifft(const T & Abar) const;

  template <std::size_t... dims>
  auto getAxisHelper(std::integer_sequence<std::size_t, dims...>) const
  {
    return std::make_tuple(getAxis(dims)...);
  }

  template <std::size_t... dims>
  auto getFrequencyHelper(std::integer_sequence<std::size_t, dims...>) const
  {
    return std::make_tuple(getFrequency(dims)...);
  }

  template <std::size_t... dims>
  auto gradHelper(std::integer_sequence<std::size_t, dims...>) const
  {
    return std::make_tuple(grad(dims)...);
  }

private:
  torch::TensorOptions _options;
  const torch::Tensor _two_pi_i;
  T _data;
  const bool _rfft;
  std::array<neml2::Real, D> _min;
  std::array<neml2::Real, D> _max;
};

template <typename T, std::size_t D>
TensorBuffer<T, D>::TensorBuffer(const TensorShapeRef & batch_shape,
                           const std::array<Real, D> & min,
                           const std::array<Real, D> & max)
  : _options(floatTensorOptions()),
    _two_pi_i(
        torch::tensor(c10::complex<double>(0.0, 2.0 * pi), MooseTensor::complexFloatTensorOptions())),
    _data(T::zeros(batch_shape, _options)),
    _rfft(true),
    _min(min),
    _max(max)
{
  if (batch_shape.size() != D)
    throw std::domain_error("Invalid dimension");
}

template <typename T, std::size_t D>
torch::Tensor
TensorBuffer<T, D>::getAxis(std::size_t dim, Interval interval) const
{
  if (dim >= D)
    throw std::domain_error("Invalid dimension");

  const auto n = _data.batch_sizes()[dim];

  switch (interval)
  {
    case Interval::OPEN:
      return torch::unsqueeze(
          torch::narrow(
              torch::linspace(c10::Scalar(_min[dim]), c10::Scalar(_max[dim]), n + 2, _options),
              0,
              1,
              n),
          D - dim - 1);

    case Interval::LEFT_OPEN:
      return torch::unsqueeze(
          torch::narrow(
              torch::linspace(c10::Scalar(_min[dim]), c10::Scalar(_max[dim]), n + 1, _options),
              0,
              1,
              n),
          D - dim - 1);

    case Interval::RIGHT_OPEN:
      return torch::unsqueeze(
          torch::narrow(
              torch::linspace(c10::Scalar(_min[dim]), c10::Scalar(_max[dim]), n + 1, _options),
              0,
              0,
              n),
          D - dim - 1);

    default: // case CLOSED:
      return torch::unsqueeze(
          torch::linspace(c10::Scalar(_min[dim]), c10::Scalar(_max[dim]), n, _options),
          D - dim - 1);
  }
}

template <typename T, std::size_t D>
torch::Tensor
TensorBuffer<T, D>::getFrequency(std::size_t dim) const
{
  if (dim >= D)
    throw std::domain_error("Invalid dimension");

  const auto n = _data.batch_sizes()[dim];
  const auto a = (_max[dim] - _min[dim]) / Real(n);
  const auto freq = (dim == D - 1 && _rfft) ? torch::fft::rfftfreq(n, a, _options)
                                            : torch::fft::fftfreq(n, a, _options);

  return torch::unsqueeze(freq, D - dim - 1);
}

template <typename T, std::size_t D>
torch::Tensor
TensorBuffer<T, D>::grad(std::size_t dim, T * cached_fft) const
{
  if (dim >= D)
    throw std::domain_error("Invalid dimension");

  if (cached_fft)
    return ifft(*cached_fft * getFrequency(dim) * _two_pi_i);
  else
    return ifft(fft() * getFrequency(dim) * _two_pi_i);
}

template <typename T, std::size_t D>
T
TensorBuffer<T, D>::fft() const
{
  if constexpr (D == 1)

    return torch::fft::rfft2(_data);

  if constexpr (D == 2)
    return torch::fft::rfft2(_data);

  if constexpr (D == 3)
    return torch::fft::irfftn(_data, c10::nullopt, {0, 1, 2});
}

template <typename T, std::size_t D>
T
TensorBuffer<T, D>::ifft(const T & Abar) const
{
  if constexpr (D == 1)

    return torch::fft::irfft(Abar);

  if constexpr (D == 2)
    return torch::fft::irfft2(Abar);

  if constexpr (D == 3)
    return torch::fft::irfftn(Abar, c10::nullopt, {0, 1, 2});
}

template <typename T, std::size_t D>
torch::Tensor
TensorBuffer<T, D>::laplace() const
{
  torch::Tensor cached_fft = fft();
  auto ret = grad(0, &cached_fft);
  ret *= ret;

  for (std::size_t dim = 1; dim < D; ++dim)
  {
    auto g = grad(dim, &cached_fft);
    ret += g * g;
  }
  return ret;
}
}
#endif
