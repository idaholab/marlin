/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorHistogram.h"
#include "MarlinUtils.h"

#include <ATen/ATen.h>
#include <ATen/native/Histogram.h>

registerMooseObject("MarlinApp", TensorHistogram);

InputParameters
TensorHistogram::validParams()
{
  InputParameters params = TensorVectorPostprocessor::validParams();
  params.addClassDescription("Compute a histogram of the given tensor.");
  params.addRequiredParam<Real>("min", "Lower bound of the histogram.");
  params.addRequiredParam<Real>("max", "Upper bound of the histogram.");
  params.addRequiredRangeCheckedParam<std::size_t>("bins", "bins>0", "Number of histogram bins.");
  return params;
}

TensorHistogram::TensorHistogram(const InputParameters & parameters)
  : TensorVectorPostprocessor(parameters),
    _min(getParam<Real>("min")),
    _max(getParam<Real>("max")),
    _bins(getParam<std::size_t>("bins")),
    _bin_edges(torch::linspace(_min, _max, _bins + 1, MooseTensor::floatTensorOptions())),
    _bin_vec(declareVector("bin")),
    _count_vec(declareVector("count"))
{
  // error check (if this is not fulfilled the histogram will be empty)
  if (_min > _max)
    paramError("min", "max must be greater than min");

  // fill the bin vector
  _bin_vec.resize(_bins);
  _count_vec.resize(_bins);
  const auto step = (_max - _min) / _bins;
  for (const auto i : make_range(_bins))
    _bin_vec[i] = _min + step / 2.0 + step * i;
}

void
TensorHistogram::execute()
{
  // Reshape the data to fit the expected input format for histogramdd
  const auto data = _u.reshape({-1, 1});

  // Use the histogramdd function
  torch::Tensor hist;
  try
  {
    // histogramdd does not have a cuda implementation in torch 2.1
    // we try anyways to run on the current compute device, in case
    // the implemntation exists for different devices or future torch versions.
    const auto pair = at::native::histogramdd(data, {_bin_edges});
    hist = std::get<0>(pair).cpu();
  }
  catch (const std::exception &)
  {
    const auto pair = at::native::histogramdd(data.cpu(), {_bin_edges.cpu()});
    hist = std::get<0>(pair);
  }

  // put into VPP vector
  if (hist.dtype() == torch::kFloat32)
    for (const auto i : make_range(int64_t(_bins)))
      _count_vec[i] = hist.index({i}).item<float>();
  else if (hist.dtype() == torch::kFloat64)
    for (const auto i : make_range(int64_t(_bins)))
      _count_vec[i] = hist.index({i}).item<double>();
  else
    mooseError("Unsupported tensor dtype() in TensorHistogram.");
}
