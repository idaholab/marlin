/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ReciprocalMeanFix.h"

registerMooseObject("MarlinApp", ReciprocalMeanFix);

InputParameters
ReciprocalMeanFix::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Fix the k=0 mode to enforce mean value in reciprocal space.");
  params.addRequiredParam<TensorInputBufferName>("input", "Input reciprocal-space buffer.");
  params.addParam<Real>("u_inf", 0.0, "Target mean value.");
  return params;
}

ReciprocalMeanFix::ReciprocalMeanFix(const InputParameters & parameters)
  : TensorOperator<>(parameters), _input(getInputBuffer("input")), _u_inf(getParam<Real>("u_inf"))
{
}

void
ReciprocalMeanFix::computeBuffer()
{
  _u = _input;

  bool has_k0 = true;
  std::vector<torch::indexing::TensorIndex> zero_idx(_dim, 0);
  if (_dim >= 1)
  {
    auto kx = _domain.getReciprocalAxis(0);
    has_k0 = has_k0 && (kx.index(zero_idx) == 0.0).item<bool>();
  }
  if (_dim >= 2)
  {
    auto ky = _domain.getReciprocalAxis(1);
    has_k0 = has_k0 && (ky.index(zero_idx) == 0.0).item<bool>();
  }
  if (_dim >= 3)
  {
    auto kz = _domain.getReciprocalAxis(2);
    has_k0 = has_k0 && (kz.index(zero_idx) == 0.0).item<bool>();
  }

  if (!has_k0)
    return;

  const auto target = _u_inf * static_cast<Real>(_domain.getNumberOfCells());

  auto value = _u.is_complex() ? torch::tensor(c10::complex<double>(target, 0.0), _u.options())
                               : torch::tensor(target, _u.options());
  if (_dim == 1)
    _u.index_put_({0}, value);
  else if (_dim == 2)
    _u.index_put_({0, 0}, value);
  else
    _u.index_put_({0, 0, 0}, value);
}
