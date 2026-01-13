/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LatticeBoltzmannOperator.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

InputParameters
LatticeBoltzmannOperator::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("LatticeBoltzmannOperator object.");
  return params;
}

LatticeBoltzmannOperator::LatticeBoltzmannOperator(const InputParameters & parameters)
  : TensorOperator(parameters),
    _lb_problem(dynamic_cast<LatticeBoltzmannProblem &>(_tensor_problem)),
    _stencil(_lb_problem.getStencil()),
    _ex(_stencil._ex.clone().reshape({1, 1, 1, _stencil._q})),
    _ey(_stencil._ey.clone().reshape({1, 1, 1, _stencil._q})),
    _ez(_stencil._ez.clone().reshape({1, 1, 1, _stencil._q})),
    _w(_stencil._weights.clone().reshape({1, 1, 1, _stencil._q})),
    _shape(_lb_problem.getExtendedShape()),
    _shape_q(_lb_problem.getExtendedShapeQ()),
    _radius(_lb_problem.getGhostRadius())
{
}

void
LatticeBoltzmannOperator::realSpaceComputeBuffer()
{
  _u_owned = _u;
  for (unsigned int d = 0; d < _dim; d++)
    _u_owned = _u_owned.narrow(d, _radius, _shape[d]);
  _lb_problem.runComputeWithGhosts(*this);
}
