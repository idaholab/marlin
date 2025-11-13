/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ReciprocalIntegral.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", ReciprocalIntegral);

InputParameters
ReciprocalIntegral::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addClassDescription("Extract the zero k-vector value (corresponding to the integral).");
  return params;
}

ReciprocalIntegral::ReciprocalIntegral(const InputParameters & parameters)
  : TensorPostprocessor(parameters)

{
}

void
ReciprocalIntegral::execute()
{
  static const at::indexing::TensorIndex zero[3] = {0, 0, 0};
  // Extract the zero-frequency component at index {0, 0, ..., 0} for arbitrary dimensions
  torch::Tensor zero_frequency_tensor =
      _u.index(torch::ArrayRef<at::indexing::TensorIndex>(zero, _u.dim()));

  // Convert to std::complex<double> (TODO: or float!)
  _integral = torch::real(zero_frequency_tensor).item<double>();

  // divide by number of cells in real space
  const auto & n = _domain.getGridSize();
  _integral /= n[0] * n[1] * n[2];

  // multiply by domain size
  const auto & volume = _domain.getVolume();
  mooseInfo(volume);
  _integral *= volume;

  // for parallelism only the proc owning the global 0 matters...
}

void
ReciprocalIntegral::finalize()
{
  // proc that owns 0 should broadcast it. everyone else listens
}

PostprocessorValue
ReciprocalIntegral::getValue() const
{
  return _integral;
}
