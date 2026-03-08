/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMConvectiveOutflow.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMConvectiveOutflow);

InputParameters
LBMConvectiveOutflow::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addClassDescription(
      "Convective outflow boundary condition. Applies df/dt + U_c * df/dn = 0 "
      "at the boundary, discretized as "
      "f(x_b, t) = (f(x_b, t-1) + U_c * f(x_n, t)) / (1 + U_c). "
      "U_c can be a fixed value or computed automatically as the mean normal "
      "velocity at the boundary plane.");

  params.addRequiredParam<TensorInputBufferName>("f_old", "Old state distribution function");

  params.addParam<std::string>(
      "convection_velocity",
      "auto",
      "Convection velocity U_c. Set to 'auto' to compute the mean normal velocity "
      "at the outlet, or provide a constant name / numeric value.");

  return params;
}

LBMConvectiveOutflow::LBMConvectiveOutflow(const InputParameters & parameters)
  : LBMBoundaryCondition(parameters),
    _f_old(_lb_problem.getBufferOld(getParam<TensorInputBufferName>("f_old"), 1)),
    _auto_velocity(getParam<std::string>("convection_velocity") == "auto"),
    _uc_value(_auto_velocity
                  ? 0.0
                  : _lb_problem.getConstant<Real>(getParam<std::string>("convection_velocity")))
{
}

torch::Tensor
LBMConvectiveOutflow::computeMeanNormalVelocity(const torch::Tensor & f_slice, int normal_component)
{
  auto rho = f_slice.sum(-1, /*keepdim=*/true);

  // Select the appropriate stencil velocity component
  torch::Tensor e_normal;
  switch (normal_component)
  {
    case 0:
      e_normal = _ex;
      break;
    case 1:
      e_normal = _ey;
      break;
    case 2:
      e_normal = _ez;
      break;
    default:
      mooseError("Invalid normal component for convective outflow");
  }

  auto u_normal = (f_slice * e_normal).sum(-1, /*keepdim=*/true) / rho;

  return torch::abs(u_normal.mean());
}

void
LBMConvectiveOutflow::applyConvectiveOutflow(int dim,
                                             int64_t b_idx,
                                             int64_t n_idx,
                                             int normal_component)
{
  // f(x_n, t): interior neighbor from the current (post-stream) distribution
  auto f_neighbor = _u_owned.select(dim, n_idx).unsqueeze(dim);

  // f(x_b, t-1): boundary node from the old state
  auto f_old_boundary = _f_old_owned.select(dim, b_idx).unsqueeze(dim);

  // convection velocity
  torch::Tensor uc;
  if (_auto_velocity)
  {
    // Compute mean normal velocity from the interior neighbor slice
    uc = computeMeanNormalVelocity(f_neighbor, normal_component);
  }
  else
    uc = torch::tensor(_uc_value, MooseTensor::floatTensorOptions());

  // Convective outflow:  f(x_b, t) = (f(x_b, t-1) + U_c * f(x_n, t)) / (1 + U_c)
  auto result = (f_old_boundary + uc * f_neighbor) / (1.0 + uc);

  _u_owned.narrow(dim, b_idx, 1).copy_(result);
}

void
LBMConvectiveOutflow::leftBoundary()
{
  // x = 0; interior neighbor at x = 1; normal is x-direction
  applyConvectiveOutflow(0, 0, 1, 0);
}

void
LBMConvectiveOutflow::rightBoundary()
{
  // x = Nx-1; interior neighbor at Nx-2; normal is x-direction
  applyConvectiveOutflow(0, _shape[0] - 1, _shape[0] - 2, 0);
}

void
LBMConvectiveOutflow::bottomBoundary()
{
  // y = 0; interior neighbor at y = 1; normal is y-direction
  applyConvectiveOutflow(1, 0, 1, 1);
}

void
LBMConvectiveOutflow::topBoundary()
{
  // y = Ny-1; interior neighbor at Ny-2; normal is y-direction
  applyConvectiveOutflow(1, _shape[1] - 1, _shape[1] - 2, 1);
}

void
LBMConvectiveOutflow::frontBoundary()
{
  // z = 0; interior neighbor at z = 1; normal is z-direction
  applyConvectiveOutflow(2, 0, 1, 2);
}

void
LBMConvectiveOutflow::backBoundary()
{
  // z = Nz-1; interior neighbor at Nz-2; normal is z-direction
  applyConvectiveOutflow(2, _shape[2] - 1, _shape[2] - 2, 2);
}

void
LBMConvectiveOutflow::computeBuffer()
{
  _f_old_owned = _f_old[0];
  for (unsigned int d = 0; d < _dim; d++)
    _f_old_owned = _f_old_owned.narrow(d, _radius, _shape[d]);

  LBMBoundaryCondition::computeBuffer();
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
