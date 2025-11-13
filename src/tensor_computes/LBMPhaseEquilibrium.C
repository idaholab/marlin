/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMPhaseEquilibrium.h"

registerMooseObject("MarlinApp", LBMPhaseEquilibrium);

InputParameters
LBMPhaseEquilibrium::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute LB equilibrium distribution object.");
  params.addRequiredParam<TensorInputBufferName>("phi", "LBM phase field parameter");
  params.addRequiredParam<TensorInputBufferName>("grad_phi",
                                                 "Gradient of LBM phase field parameter");
  params.addRequiredParam<std::string>("tau_phi", "Relaxation parameter for LBM phase field");
  params.addRequiredParam<std::string>("thickness", "Interface thickness");

  return params;
}

LBMPhaseEquilibrium::LBMPhaseEquilibrium(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _phi(getInputBuffer("phi")),
    _grad_phi(getInputBuffer("grad_phi")),
    _tau_phi(_lb_problem.getConstant<Real>(getParam<std::string>("tau_phi"))),
    _D(_lb_problem.getConstant<Real>(getParam<std::string>("thickness")))
{
}

void
LBMPhaseEquilibrium::computeBuffer()
{
  const unsigned int & dim = _domain.getDim();

  if (_phi.dim() < 3)
    _phi.unsqueeze_(2);

  torch::Tensor _phi_unsqueezed = _phi.unsqueeze(3);

  // in the future when phase field is coupled with NS this will be extended to include fluid
  // velocity and density
  auto gamma_eq = _w * _phi_unsqueezed;

  switch (dim)
  {
    case 3:
      mooseError("Not implemented fo 3D yet!");
      break;
    case 2:
    {
      torch::Tensor phase_eq_2;
      {
        torch::Tensor phase_eq;
        torch::Tensor e_dot_n;
        {
          auto mag = torch::norm(_grad_phi, 2, -1);
          // _lb_problem.printBuffer(mag, 10, 0);

          auto unit_normal = _grad_phi / (mag.unsqueeze(-1) + 1.0e-16);
          unit_normal.unsqueeze_(3);
          // _lb_problem.printBuffer(unit_normal, 10, 0);

          auto e_xyz = torch::stack(
                           {
                               _ex,
                               _ey,
                           },
                           -1)
                           .to(MooseTensor::floatTensorOptions());
          e_dot_n = torch::einsum("ijklm,abcdm->abcl", {e_xyz, unit_normal});
          // _lb_problem.printBuffer(e_dot_n, 10, 1);
          phase_eq = 4.0 / _D * _phi_unsqueezed * (1.0 - _phi_unsqueezed) * e_dot_n;
        }
        phase_eq_2 = _w * (_tau_phi)*phase_eq;
      }
      _u = gamma_eq; // + phase_eq_2;
      // _lb_problem.printBuffer(_u, 10, 1);
      break;
    }
    default:
      mooseError("Unsupported dimension for buffer _u");
  }
  _lb_problem.maskedFillSolids(_u, 0);
}
