/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FCCPFCLinear.h"

#include <cmath>

registerMooseObject("MarlinApp", FCCPFCLinear);

InputParameters
FCCPFCLinear::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription(
      "Reciprocal space linear prefactor for the two-mode FCC phase-field crystal model.");
  params.addParam<Real>("eps", -0.5, "Undercooling parameter.");
  params.addParam<Real>("Q1", std::sqrt(4.0 / 3.0), "First mode wave number.");
  params.addParam<Real>("R1", 0.0, "Second-mode strength (>= 0).");
  params.addParam<Real>("mobility", 1.0, "Mobility prefactor.");
  return params;
}

FCCPFCLinear::FCCPFCLinear(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _eps(getParam<Real>("eps")),
    _q1(getParam<Real>("Q1")),
    _r1(getParam<Real>("R1")),
    _mobility(getParam<Real>("mobility")),
    _k2(_domain.getKSquare())
{
  if (_r1 < 0.0)
    paramError("R1", "R1 must be non-negative.");
}

void
FCCPFCLinear::computeBuffer()
{
  const auto one_minus_k2 = 1.0 - _k2;
  const auto q1_sq_minus_k2 = _q1 * _q1 - _k2;
  const auto lhat = -_eps + one_minus_k2 * one_minus_k2 * (q1_sq_minus_k2 * q1_sq_minus_k2 + _r1);

  _u = -_k2 * lhat * _mobility;
}
