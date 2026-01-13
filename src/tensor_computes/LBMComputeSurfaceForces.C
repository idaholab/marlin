/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMComputeSurfaceForces.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMComputeSurfaceForces);

InputParameters
LBMComputeSurfaceForces::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();

  params.addRequiredParam<TensorInputBufferName>("chemical_potential",
                                                 "Macroscopic chemical potential");
  params.addRequiredParam<TensorInputBufferName>("grad_phi", "Gradient of phase field");

  params.addClassDescription("Compute object for LB surface forces");
  return params;
}

LBMComputeSurfaceForces::LBMComputeSurfaceForces(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _chemical_potential(getInputBuffer("chemical_potential", _radius)),
    _grad_phi(getInputBuffer("grad_phi", _radius))
{
}

void
LBMComputeSurfaceForces::computeBuffer()
{
  _u = _chemical_potential.unsqueeze(-1) * _grad_phi;
  _lb_problem.maskedFillSolids(_u, 0);
}
