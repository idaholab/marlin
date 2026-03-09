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
#include "DomainAction.h"

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

void
LBMConvectiveOutflow::precomputeConvectionVelocity()
{
  Real local_sum = 0.0;
  Real local_count = 0.0;

  // Only boundary-owning ranks contribute to the sum
  const uint8_t bit = (1 << static_cast<int>(_boundary));
  if (_boundary_rank & bit)
  {
    int dim = -1, normal_component = -1;
    int64_t n_idx = -1;
    switch (_boundary)
    {
      case Boundary::left:
        dim = 0;
        n_idx = 1;
        normal_component = 0;
        break;
      case Boundary::right:
        dim = 0;
        n_idx = _shape[0] - 2;
        normal_component = 0;
        break;
      case Boundary::bottom:
        dim = 1;
        n_idx = 1;
        normal_component = 1;
        break;
      case Boundary::top:
        dim = 1;
        n_idx = _shape[1] - 2;
        normal_component = 1;
        break;
      case Boundary::front:
        dim = 2;
        n_idx = 1;
        normal_component = 2;
        break;
      case Boundary::back:
        dim = 2;
        n_idx = _shape[2] - 2;
        normal_component = 2;
        break;
      default:
        mooseError("Unsupported boundary for convective outflow auto velocity");
    }

    auto f_neighbor = _u_owned.select(dim, n_idx).unsqueeze(dim);
    auto rho = f_neighbor.sum(-1, /*keepdim=*/true);

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

    auto u_normal = (f_neighbor * e_normal).sum(-1, /*keepdim=*/true) / rho;
    local_sum = torch::abs(u_normal).sum().item<Real>();
    local_count = static_cast<Real>(u_normal.numel());
  }

  // Global reduction - ALL ranks participate regardless of boundary ownership
  _domain.comm().sum(local_sum);
  _domain.comm().sum(local_count);

  _uc_computed = torch::tensor(local_sum / local_count, MooseTensor::floatTensorOptions());
}

void
LBMConvectiveOutflow::applyConvectiveOutflow(int dim, int64_t b_idx, int64_t n_idx)
{
  // f(x_n, t): interior neighbor from the current (post-stream) distribution
  auto f_neighbor = _u_owned.select(dim, n_idx).unsqueeze(dim);

  // f(x_b, t-1): boundary node from the old state
  auto f_old_boundary = _f_old_owned.select(dim, b_idx).unsqueeze(dim);

  // convection velocity (pre-computed in computeBuffer for auto, constant otherwise)
  torch::Tensor uc =
      _auto_velocity ? _uc_computed : torch::tensor(_uc_value, MooseTensor::floatTensorOptions());

  // Convective outflow:  f(x_b, t) = (f(x_b, t-1) + U_c * f(x_n, t)) / (1 + U_c)
  auto result = (f_old_boundary + uc * f_neighbor) / (1.0 + uc);

  _u_owned.narrow(dim, b_idx, 1).copy_(result);
}

void
LBMConvectiveOutflow::leftBoundary()
{
  // x = 0; interior neighbor at x = 1
  applyConvectiveOutflow(0, 0, 1);
}

void
LBMConvectiveOutflow::rightBoundary()
{
  // x = Nx-1; interior neighbor at Nx-2
  applyConvectiveOutflow(0, _shape[0] - 1, _shape[0] - 2);
}

void
LBMConvectiveOutflow::bottomBoundary()
{
  // y = 0; interior neighbor at y = 1
  applyConvectiveOutflow(1, 0, 1);
}

void
LBMConvectiveOutflow::topBoundary()
{
  // y = Ny-1; interior neighbor at Ny-2
  applyConvectiveOutflow(1, _shape[1] - 1, _shape[1] - 2);
}

void
LBMConvectiveOutflow::frontBoundary()
{
  // z = 0; interior neighbor at z = 1
  applyConvectiveOutflow(2, 0, 1);
}

void
LBMConvectiveOutflow::backBoundary()
{
  // z = Nz-1; interior neighbor at Nz-2
  applyConvectiveOutflow(2, _shape[2] - 1, _shape[2] - 2);
}

void
LBMConvectiveOutflow::computeBuffer()
{
  _f_old_owned = _f_old[0];
  for (unsigned int d = 0; d < _dim; d++)
    _f_old_owned = _f_old_owned.narrow(d, _radius, _shape[d]);

  // Pre-compute convection velocity with global MPI reduction BEFORE the
  // parent dispatches to boundary-specific methods. This ensures ALL ranks
  // participate in the collective, avoiding deadlock when only a subset of
  // ranks own the boundary.
  if (_auto_velocity)
    precomputeConvectionVelocity();

  LBMBoundaryCondition::computeBuffer();
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
