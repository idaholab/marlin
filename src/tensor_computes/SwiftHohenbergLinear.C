/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SwiftHohenbergLinear.h"

registerMooseObject("MarlinApp", SwiftHohenbergLinear);

InputParameters
SwiftHohenbergLinear::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Reciprocal space linear term in the semi-implicit time integration "
                             "of the Swift-Hohenberg equation IC.");
  params.addParam<Real>("r", -0.5, "Phase field crystal parameter r");
  params.addParam<Real>("alpha", 1.0, "Regularization factor <=1");
  return params;
}

SwiftHohenbergLinear::SwiftHohenbergLinear(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _r(getParam<Real>("r")),
    _alpha(getParam<Real>("alpha")),
    _k2(_domain.getKSquare())
{
}

void
SwiftHohenbergLinear::computeBuffer()
{
  _u = (_r - _alpha * _alpha * (1.0 - _k2) * (1.0 - _k2));
}
