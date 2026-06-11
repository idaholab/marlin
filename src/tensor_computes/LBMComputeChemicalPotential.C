/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMComputeChemicalPotential.h"

registerMooseObject("MarlinApp", LBMComputeChemicalPotential);

InputParameters
LBMComputeChemicalPotential::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute LB checmial potential for pahse field coupling.");
  params.addRequiredParam<TensorInputBufferName>("phi", "Phase field order parameter");
  params.addRequiredParam<TensorInputBufferName>("laplacian_phi",
                                                 "Laplacian of phase field order parameter");
  params.addRequiredParam<std::string>("thickness", "Interface thickness");
  params.addRequiredParam<std::string>("sigma", "Interfacial tension coefficient");
  return params;
}

LBMComputeChemicalPotential::LBMComputeChemicalPotential(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _phi(getInputBuffer("phi", _radius)),
    _laplacian_phi(getInputBuffer("laplacian_phi", _radius)),
    _D(_lb_problem.getConstant<Real>(getParam<std::string>("thickness"))),
    _sigma(_lb_problem.getConstant<Real>(getParam<std::string>("sigma")))
{
}

void
LBMComputeChemicalPotential::computeBuffer()
{
  const int64_t N = _u.numel();
  auto u_flat = _u.view({N});
  auto phi_flat = _phi.view({N});
  auto lap_flat = _laplacian_phi.view({N});

  u_flat.copy_(phi_flat);                // u = phi
  u_flat.square_();                      // u = phi^2
  u_flat.add_(phi_flat, /*alpha=*/-1.5); // u = phi^2 - 1.5*phi
  u_flat.mul_(phi_flat);                 // u = phi^3 - 1.5*phi^2
  u_flat.add_(phi_flat, /*alpha=*/0.5);  // u = phi^3 - 1.5*phi^2 + 0.5*phi

  u_flat.mul_(48.0 * _sigma / _D);                    // Scale by 48*sigma/D
  u_flat.sub_(lap_flat, /*alpha=*/1.5 * _D * _sigma); // u = u - 1.5*D*sigma*laplacian_phi

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
