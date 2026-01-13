/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMMicroscopicZeroGradientBC.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMMicroscopicZeroGradientBC);

InputParameters
LBMMicroscopicZeroGradientBC::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addClassDescription("LBMMicroscopicZeroGradientBC object");
  return params;
}

LBMMicroscopicZeroGradientBC::LBMMicroscopicZeroGradientBC(const InputParameters & parameters)
  : LBMBoundaryCondition(parameters)
{
}

void
LBMMicroscopicZeroGradientBC::leftBoundary()
{
  _u.index_put_({0, Slice(), Slice(), Slice()}, _u.index({1, Slice(), Slice(), Slice()}));
}

void
LBMMicroscopicZeroGradientBC::rightBoundary()
{
  _u.index_put_({_shape[0] - 1, Slice(), Slice(), Slice()},
                _u.index({_shape[0] - 2, Slice(), Slice(), Slice()}));
}

void
LBMMicroscopicZeroGradientBC::bottomBoundary()
{
  _u.index_put_({Slice(), 0, Slice(), Slice()}, _u.index({Slice(), 1, Slice(), Slice()}));
}

void
LBMMicroscopicZeroGradientBC::topBoundary()
{
  _u.index_put_({Slice(), _shape[1] - 1, Slice(), Slice()},
                _u.index({Slice(), _shape[1] - 2, Slice(), Slice()}));
}

void
LBMMicroscopicZeroGradientBC::frontBoundary()
{
  _u.index_put_({Slice(), Slice(), 0, Slice()}, _u.index({Slice(), Slice(), 1, Slice()}));
}

void
LBMMicroscopicZeroGradientBC::backBoundary()
{
  _u.index_put_({Slice(), Slice(), _shape[2] - 1, Slice()},
                _u.index({Slice(), Slice(), _shape[2] - 2, Slice()}));
}

void
LBMMicroscopicZeroGradientBC::computeBuffer()
{
  // do not overwrite previous
  _u = _u.clone();
  LBMBoundaryCondition::computeBuffer();
  _lb_problem.maskedFillSolids(_u, 0);
}
