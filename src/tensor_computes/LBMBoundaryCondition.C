/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMBoundaryCondition.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

InputParameters
LBMBoundaryCondition::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  MooseEnum boundary("top bottom left right front back wall regional");
  params.addRequiredParam<MooseEnum>(
      "boundary", boundary, "Edges/Faces where boundary condition is applied.");
  params.addClassDescription("LBMBoundaryCondition object.");
  return params;
}

LBMBoundaryCondition::LBMBoundaryCondition(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _grid_size(_lb_problem.getGridSize()),
    _boundary(getParam<MooseEnum>("boundary").getEnum<Boundary>())
{
}

void
LBMBoundaryCondition::computeBuffer()
{
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
      wallBoundary();
      break;
    case Boundary::regional:
      regionalBoundary();
      break;
    default:
      mooseError("Undefined boundary names");
  }
  _lb_problem.maskedFillSolids(_u, 0);
}
