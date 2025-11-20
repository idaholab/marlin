/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ReciprocalLaplacianFactor.h"

registerMooseObject("MarlinApp", ReciprocalLaplacianFactor);

InputParameters
ReciprocalLaplacianFactor::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Reciprocal space Laplacian IC.");
  params.addParam<Real>("factor", 1.0, "Prefactor");
  return params;
}

ReciprocalLaplacianFactor::ReciprocalLaplacianFactor(const InputParameters & parameters)
  : TensorOperator<>(parameters), _factor(getParam<Real>("factor")), _k2(_domain.getKSquare())
{
}

void
ReciprocalLaplacianFactor::computeBuffer()
{
  _u = -_k2 * _factor;
}
