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
  _u.index_put_({_grid_size[0] - 1, Slice(), Slice(), Slice()},
                _u.index({_grid_size[0] - 2, Slice(), Slice(), Slice()}));
}

void
LBMMicroscopicZeroGradientBC::bottomBoundary()
{
  _u.index_put_({Slice(), 0, Slice(), Slice()}, _u.index({Slice(), 1, Slice(), Slice()}));
}

void
LBMMicroscopicZeroGradientBC::topBoundary()
{
  _u.index_put_({Slice(), _grid_size[1] - 1, Slice(), Slice()},
                _u.index({Slice(), _grid_size[1] - 2, Slice(), Slice()}));
}

void LBMMicroscopicZeroGradientBC::frontBoundary()
{
  _u.index_put_({Slice(), Slice(), 0, Slice()}, _u.index({Slice(), Slice(), 1, Slice()}));
}

void LBMMicroscopicZeroGradientBC::backBoundary()
{
  _u.index_put_({Slice(), Slice(), _grid_size[2] - 1, Slice()},
                _u.index({Slice(), Slice(), _grid_size[2] - 2, Slice()}));
}

void
LBMMicroscopicZeroGradientBC::computeBuffer()
{
  // do not overwrite previous
  _u = _u.clone();

  switch (_boundary)
  {
    case Boundary::top:
      topBoundary();
      break;
    case Boundary::bottom:
      bottomBoundary();
      break;
    case Boundary::left:
      leftBoundary();
      break;
    case Boundary::right:
      rightBoundary();
      break;
    case Boundary::front:
      frontBoundary();
      break;
    case Boundary::back:
      backBoundary();
      break;
    case Boundary::wall:
      mooseError("Wall boundary is not implemented");
      break;
    default:
      mooseError("Undefined boundary names");
  }
  _lb_problem.maskedFillSolids(_u, 0);
}
