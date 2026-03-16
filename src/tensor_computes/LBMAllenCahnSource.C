/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMAllenCahnSource.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMAllenCahnSource);

InputParameters
LBMAllenCahnSource::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute Allen-Cahn source term for phase field model.");
  params.addRequiredParam<TensorInputBufferName>("phi",
                                                 "LBM phase field parameter");
  params.addRequiredParam<TensorInputBufferName>("velocity",
                                                 "LBM fluid velocity");
  params.addRequiredParam<TensorInputBufferName>("grad_phi",
                                                 "Gradient of LBM phase field parameter");
  params.addRequiredParam<std::string>("tau", "Relaxation parameter for LBM phase field");
  params.addRequiredParam<std::string>("thickness", "Interface thickness");

  return params;
}

LBMAllenCahnSource::LBMAllenCahnSource(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _phi(getInputBuffer("phi", _radius)),
    _velocity(getInputBuffer("velocity", _radius)),
    _grad_phi(getInputBuffer("grad_phi", _radius)),
    _tau(_lb_problem.getConstant<Real>(getParam<std::string>("tau"))),
    _D(_lb_problem.getConstant<Real>(getParam<std::string>("thickness")))
{
  std::vector<int64_t> shape_q_with_ghost = _shape_q;
  shape_q_with_ghost[0] += 2 * _radius;
  shape_q_with_ghost[1] += 2 * _radius;
  if (_dim == 3)
    shape_q_with_ghost[2] += 2 * _radius;
  _source_term = torch::zeros(shape_q_with_ghost, MooseTensor::floatTensorOptions());
}

void
LBMAllenCahnSource::computeSourceTerm()
{
  const unsigned int & dim = _domain.getDim();

  // Lazily initialize _phi_u_old on first call
  if (_phi_u_old.numel() == 0)
    _phi_u_old = torch::zeros_like(_velocity);

  if (_phi.dim() < 3)
    _phi.unsqueeze_(2);

  torch::Tensor phi_unsqueezed = _phi.unsqueeze(-1);
  torch::Tensor phi_u = phi_unsqueezed * _velocity;
  torch::Tensor dphi_u_dt = phi_u - _phi_u_old;

  // Unit normal from gradient of phi
  auto mag = torch::norm(_grad_phi, 2, -1);
  auto unit_normal = _grad_phi / (mag.unsqueeze(-1) + 1.0e-16);
  
  // Anti-diffusion coefficient: lambda = 4*phi*(1-phi)/D
  torch::Tensor lambda = 4.0 / _D * phi_unsqueezed * (1.0 - phi_unsqueezed);
  // Combined vector: A = d(phi*u)/dt + cs2 * lambda * n
  torch::Tensor A = dphi_u_dt + _lb_problem._cs2 * lambda * unit_normal;

  torch::Tensor Ax = A.select(3, 0).unsqueeze(-1);
  torch::Tensor Ay = A.select(3, 1).unsqueeze(-1);
  torch::Tensor Az;

  switch (dim)
  {
    case 3:
      Az = A.select(3, 2).unsqueeze(-1);
      break;
    case 2:
      Az = torch::zeros_like(Ax);
      break;
    default:
      mooseError("Unsupported dimension for LBMAllenCahnSource");
  }
  // source term: w_i * (c_i . A) / cs2
  for (int64_t ic = 0; ic < _stencil._q; ic++)
  {
    _source_term.index_put_(
        {Slice(), Slice(), Slice(), ic},
        _stencil._weights[ic] *
            (_stencil._ex[ic] * Ax + _stencil._ey[ic] * Ay + _stencil._ez[ic] * Az).squeeze(-1) /
            _lb_problem._cs2);
  }

  // Store current phi*u for next timestep
  _phi_u_old.copy_(phi_u);
}

void
LBMAllenCahnSource::computeBuffer()
{
  computeSourceTerm();
  _u += (1.0 - 1.0 / (2.0 * _tau)) * _source_term;
  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
