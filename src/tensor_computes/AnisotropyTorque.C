/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "AnisotropyTorque.h"

registerMooseObject("MarlinApp", AnisotropyTorque);

InputParameters
AnisotropyTorque::validParams()
{
  InputParameters params = FFTGradientBase<>::validParams();
  params.addClassDescription(
      "Computes the anisotropy torque as the sum of FFT gradient components of dmu_dn * g.");
  params.addRequiredParam<TensorInputBufferName>(
      "dmu_dn", "Input tensor buffer containing derivatives with respect to the normal tensor.");
  params.addRequiredParam<TensorInputBufferName>("g", "Scalar weighting tensor buffer.");
  return params;
}

AnisotropyTorque::AnisotropyTorque(const InputParameters & parameters)
  : FFTGradientBase<>(parameters),
    _dmu_dn(getInputBuffer("dmu_dn")),
    _g(getInputBuffer("g"))
{
}

void
AnisotropyTorque::computeBuffer()
{
  if (_dmu_dn.dim() < 1)
    mooseError("dmu_dn must have at least one dimension.");

  if (_dmu_dn.size(-1) != 3)
    mooseError("dmu_dn must have trailing component dimension of size 3.");

  if (_g.sizes() != _dmu_dn.sizes().slice(0, _dmu_dn.dim() - 1))
    mooseError("g must have the same spatial shape as dmu_dn without its trailing component axis.");

  _u = torch::zeros_like(_g);
  for (const auto i : make_range(_dim))
  {
    auto weighted_component = _dmu_dn.select(-1, i) * _g;
    _u += computeGradientComponent(weighted_component, /* input_is_reciprocal = */ false, i);
  }
}
