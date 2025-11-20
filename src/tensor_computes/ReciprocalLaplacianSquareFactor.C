/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ReciprocalLaplacianSquareFactor.h"

registerMooseObject("MarlinApp", ReciprocalLaplacianSquareFactor);

InputParameters
ReciprocalLaplacianSquareFactor::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Reciprocal space Laplacian squared IC.");
  params.addParam<Real>("factor", 1.0, "Prefactor");
  return params;
}

ReciprocalLaplacianSquareFactor::ReciprocalLaplacianSquareFactor(const InputParameters & parameters)
  : TensorOperator<>(parameters), _factor(getParam<Real>("factor")), _k2(_domain.getKSquare())
{
}

void
ReciprocalLaplacianSquareFactor::computeBuffer()
{
  // ignore the minus which would drop in the next step anyways
  _u = _k2 * _k2 * _factor;
}
