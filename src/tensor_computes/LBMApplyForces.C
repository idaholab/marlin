/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
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
  params.addParam<TensorInputBufferName>("velocity", "u", "Macroscopic velocity");
  params.addRequiredParam<TensorInputBufferName>("rho", "Macroscopic density");
  params.addRequiredParam<TensorInputBufferName>("forces", "LBM forces");
  params.addRequiredParam<std::string>("tau0", "Relaxation parameter");
  params.addClassDescription("Compute object for LB forces");
  return params;
}

LBMApplyForces::LBMApplyForces(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _velocity(getInputBuffer("velocity", _radius)),
    _density(getInputBuffer("rho", _radius)),
    _forces(getInputBuffer("forces", _radius)),
    _tau(_lb_problem.getConstant<Real>(getParam<std::string>("tau0")))
{
  _shape_q_with_ghost = _shape_q;
  _shape_q_with_ghost[0] += 2 * _radius;
  _shape_q_with_ghost[1] += 2 * _radius;
  if (_dim == 3)
    _shape_q_with_ghost[2] += 2 * _radius;

  _source_term = torch::zeros(_shape_q_with_ghost, MooseTensor::floatTensorOptions());

  // Precompute the force projection matrix: _P_mat = E * (w / cs2)
  std::vector<torch::Tensor> e_vec = {_stencil._ex, _stencil._ey};
  if (_dim == 3)
    e_vec.push_back(_stencil._ez);

  torch::Tensor E_mat =
      torch::stack(e_vec, /*dim=*/0).to(MooseTensor::floatTensorOptions()); // [dim, Q]
  auto w_flat = _stencil._weights.unsqueeze(0);                             // [1, Q]

  _P_mat = (E_mat * (w_flat / _lb_problem._cs2)).clone();
}

void
LBMApplyForces::computeSourceTerm()
{
  const unsigned int dim = _domain.getDim();
  const int64_t N = _density.numel();
  const int Q = _stencil._q;

  auto rho_flat = _density.view({N, 1});
  auto F_flat = _forces.slice(-1, 0, dim).reshape({N, dim});
  auto source_flat = _source_term.view({N, Q});

  // [N, dim] @ [dim, Q]
  torch::mm_out(source_flat, F_flat, _P_mat);
  source_flat.mul_(rho_flat);
}

void
LBMApplyForces::computeBuffer()
{
  computeSourceTerm();

  const int64_t N = _u.numel() / _stencil._q;
  auto u_flat = _u.view({N, _stencil._q});
  auto source_flat = _source_term.view({N, _stencil._q});

  u_flat.add_(source_flat, /*alpha=*/1.0 - 1.0 / (2.0 * _tau));

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
