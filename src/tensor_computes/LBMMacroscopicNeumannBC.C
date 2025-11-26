/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMMacroscopicNeumannBC.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMMacroscopicNeumannBC);

InputParameters
LBMMacroscopicNeumannBC::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addClassDescription(
      "LBMMacroscopicNeumannBC object that uses first order forward difference");
  params.addParam<std::string>("value",
                               "0.0"
                               "Value at the boundary");
  return params;
}

LBMMacroscopicNeumannBC::LBMMacroscopicNeumannBC(const InputParameters & parameters)
  : LBMBoundaryCondition(parameters),
    _value(_lb_problem.getConstant<Real>(getParam<std::string>("value")))
{
}

void
LBMMacroscopicNeumannBC::bottomBoundary()
{
  // along y direction at y = 0
  _u.index_put_({Slice(), 0, Slice()}, _u.index({Slice(), 1, Slice()}) - _value);
}

void
LBMMacroscopicNeumannBC::topBoundary()
{
  // along y direction at y = N_y - 1
  _u.index_put_({Slice(), _grid_size[1] - 1, Slice()},
                _u.index({Slice(), _grid_size[1] - 2, Slice()}) + _value);
}

void
LBMMacroscopicNeumannBC::leftBoundary()
{
  // along x direction at x = 0
  _u.index_put_({0, Slice(), Slice()}, _u.index({1, Slice(), Slice()}) - _value);
}

void
LBMMacroscopicNeumannBC::rightBoundary()
{
  // along x direction at x = N_x - 1
  _u.index_put_({_grid_size[0] - 1, Slice(), Slice()},
                _u.index({_grid_size[1] - 2, Slice(), Slice()}) + _value);
}

void
LBMMacroscopicNeumannBC::frontBoundary()
{
  // along z direction at z = 0
  _u.index_put_({Slice(), Slice(), 0}, _u.index({Slice(), Slice(), 1}) - _value);
}

void
LBMMacroscopicNeumannBC::backBoundary()
{
  // along z direction at z = N_z-1
  _u.index_put_({Slice(), Slice(), _grid_size[2] - 1},
                _u.index({Slice(), Slice(), _grid_size[2] - 2}) + _value);
}

void
LBMMacroscopicNeumannBC::wallBoundary()
{
  // TBD
  mooseError("LBMMacroscopicNeumannBC::wallBoundary is not implemented");
}
