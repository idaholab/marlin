/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMComputePhysicalPressure.h"
#include "LatticeBoltzmannProblem.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", LBMComputePhysicalPressure);

InputParameters
LBMComputePhysicalPressure::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Computes physical gauge pressure based on density and cell size.");
  params.addRequiredParam<TensorInputBufferName>("rho", "LBM density");
  params.addRequiredParam<Real>("rho0", "LBM reference density (typically initial mean density).");
  params.addRequiredParam<Real>("rho0_phys",
                                "Physical reference density (e.g. 1000 kg/m^3 for water).");
  params.addRequiredParam<std::string>("tau", "Relaxation parameter");
  params.addRequiredParam<Real>("nu", "Physical fluid kinematic viscosity.");
  return params;
}

LBMComputePhysicalPressure::LBMComputePhysicalPressure(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _rho(getInputBuffer("rho", _radius)),
    _rho0(getParam<Real>("rho0")),
    _rho0_phys(getParam<Real>("rho0_phys")),
    _tau(_tensor_problem.getConstant<Real>(getParam<std::string>("tau"))),
    _nu(getParam<Real>("nu")),
    _grid_spacing(_domain.getGridSpacing())
{
}

void
LBMComputePhysicalPressure::computeBuffer()
{
  // Get the delta_x by averaging the grid spacing
  const unsigned int & dim = _domain.getDim();
  Real cx = 0.0;
  for (const auto i : libMesh::make_range(dim))
    cx += _grid_spacing(i) / (Real)dim;

  // delta_t can be deduced from kinematic viscosity and tau
  const Real ct = _lb_problem._cs2 / _nu * (_tau - 0.5) * cx * cx;

  // Kp = rho_0 C_s^2 (delta_x / delta_y)^2
  const Real factor = _rho0_phys * _lb_problem._cs2 * cx * cx / (ct * ct);

  // P_gauge = Kp (rho^* - rho0^*)
  _u = factor * (_rho - _rho0);

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
