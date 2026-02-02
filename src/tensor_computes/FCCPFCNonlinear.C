/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FCCPFCNonlinear.h"

registerMooseObject("MarlinApp", FCCPFCNonlinear);

InputParameters
FCCPFCNonlinear::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription(
      "Reciprocal space nonlinear term for the two-mode FCC phase-field crystal model.");
  params.addRequiredParam<TensorInputBufferName>("psi", "Phase field variable buffer.");
  params.addParam<TensorInputBufferName>("dealiasing", "Optional de-aliasing filter buffer.");
  params.addParam<Real>("mobility", 1.0, "Mobility prefactor.");
  return params;
}

FCCPFCNonlinear::FCCPFCNonlinear(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _psi(getInputBuffer("psi")),
    _dealiasing(isParamValid("dealiasing") ? &getInputBuffer("dealiasing") : nullptr),
    _mobility(getParam<Real>("mobility")),
    _k2(_domain.getKSquare())
{
}

void
FCCPFCNonlinear::computeBuffer()
{
  auto psi3_hat = _domain.fft(_psi * _psi * _psi);
  if (_dealiasing)
    psi3_hat = psi3_hat * *_dealiasing;

  _u = -_k2 * psi3_hat * _mobility;
}
