/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMApplyForces.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMApplyForces);

InputParameters
LBMApplyForces::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  // params.addRequiredParam<TensorInputBufferName>("f", "Distribution function");
  params.addParam<TensorInputBufferName>("velocity", "u", "Macroscopic velocity");
  params.addRequiredParam<TensorInputBufferName>("rho", "Macroscopic density");
  params.addRequiredParam<TensorInputBufferName>("forces", "LBM forces");
  params.addRequiredParam<std::string>("tau0", "Relaxation parameter");
  params.addClassDescription("Compute object for LB forces");
  return params;
}

LBMApplyForces::LBMApplyForces(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters), /*_f(getInputBuffer("f"))*/
    _velocity(getInputBuffer("velocity")),
    _density(getInputBuffer("rho")),
    _forces(getInputBuffer("forces")),
    _tau(_lb_problem.getConstant<Real>(getParam<std::string>("tau0")))
{
}

void
LBMApplyForces::computeSourceTerm()
{
  const unsigned int & dim = _domain.getDim();

  if (_density.dim() < 3)
    _density.unsqueeze_(2);

  torch::Tensor rho_unsqueezed = _density.unsqueeze(-1);
  torch::Tensor Fx = _forces.select(3, 0).unsqueeze(3);
  torch::Tensor Fy = _forces.select(3, 1).unsqueeze(3);
  torch::Tensor Fz;

  // torch::Tensor ux = _velocity.select(3, 0).unsqueeze(3);
  // torch::Tensor uy = _velocity.select(3, 1).unsqueeze(3);
  // torch::Tensor uz;

  torch::Tensor e_xyz = torch::stack({_stencil._ex, _stencil._ey, _stencil._ez}, 0);

  switch (dim)
  {
    case 3:
    {
      Fz = _forces.select(3, 2).unsqueeze(3);
      // uz = _velocity.select(3, 2).unsqueeze(3);
      break;
    }
    case 2:
    {
      Fz = torch::zeros_like(rho_unsqueezed, MooseTensor::floatTensorOptions());
      // uz = torch::zeros_like(rho_unsqueezed, MooseTensor::floatTensorOptions());
      break;
    }
    default:
      mooseError("Unsupported dimensions for buffer _u");
  }

  // torch::Tensor Fxyz = torch::stack({Fx, Fy, Fz}, 3).squeeze(-1);
  // torch::Tensor Uxyz = torch::stack({ux, uy, uz}, 3).squeeze(-1);
  // torch::Tensor Fxyz_expanded = Fxyz.unsqueeze(-1);       // Shape: (Nx, Ny, Nz, 3, 1)
  // torch::Tensor Uxyz_expanded = Uxyz.unsqueeze(-2);       // Shape: (Nx, Ny, Nz, 1, 3)
  // torch::Tensor UF_outer = Fxyz_expanded * Uxyz_expanded; // Shape: (Nx, Ny, Nz, 3, 3)
  // torch::Tensor UF_outer_flat = UF_outer.flatten(-2, -1); // Shape: (Nx, Ny, Nz, 9)

  for (int64_t ic = 0; ic < _stencil._q; ic++)
  {
    // auto exyz_ic = e_xyz.index({Slice(), ic}).flatten(); // Shape (3)
    // torch::Tensor ccr = torch::outer(exyz_ic, exyz_ic) / _lb_problem._cs2 -
    //                     torch::eye(3, MooseTensor::floatTensorOptions()); // Shape (9)
    // auto ccr_flat = ccr.flatten();
    // torch::Tensor multiplied = UF_outer_flat * ccr_flat; // Shape: (Nx, Ny, Nz, 9)

    // // sum along the last dimension
    // torch::Tensor UFccr = multiplied.sum(/*dim=*/-1);

    // compute source
    _source_term.index_put_(
        {Slice(), Slice(), Slice(), ic},
        _stencil._weights[ic] * rho_unsqueezed.squeeze(-1) *
            ((_stencil._ex[ic] * Fx + _stencil._ey[ic] * Fy + _stencil._ez[ic] * Fz).squeeze(-1) /
             _lb_problem._cs2 /* + UFccr / 2.0 / _lb_problem._cs4 */));
  }
}

void
LBMApplyForces::computeBuffer()
{
  computeSourceTerm();
  _u += (1.0 - 1.0 / (2.0 * _tau)) * _source_term;
  _lb_problem.maskedFillSolids(_u, 0);
}
