/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMComputeEffectiveRelaxation.h"

registerMooseObject("MarlinApp", LBMComputeEffectiveRelaxation);

using torch::indexing::Slice;

InputParameters
LBMComputeEffectiveRelaxation::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();

  params.addRequiredParam<TensorInputBufferName>("local_pore_size", "Local pore size");
  params.addRequiredParam<TensorInputBufferName>("local_Knudsen_number", "Local Knudsen number");
  params.addParam<std::string>("mfp", "1.0e-9", "Mean free path of the system, (meters)");
  params.addParam<std::string>("dx", "1.0e-9", "Domain resolution, (meters)");
  params.addParam<std::string>("A2", "0.8", "Second order slip boundary constant");

  params.addClassDescription("Compute local effective relaxation time matrix based on local pore "
                             "size and Knudsen number.");

  return params;
}

LBMComputeEffectiveRelaxation::LBMComputeEffectiveRelaxation(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _local_pore_size(getInputBuffer("local_pore_size", _radius)),
    _local_Knudsen_number(getInputBuffer("local_Knudsen_number", _radius)),
    _mfp(_lb_problem.getConstant<Real>(getParam<std::string>("mfp"))),
    _dx(_lb_problem.getConstant<Real>(getParam<std::string>("dx"))),
    _A2(_lb_problem.getConstant<Real>(getParam<std::string>("A2")))
{
  if (_stencil._q != 9)
    mooseError("LBMComputeEffectiveRelaxation currently only supports D2Q9 stencils.");
}

void
LBMComputeEffectiveRelaxation::computeBuffer()
{
  auto tau_s = 0.5 + std::sqrt(6.0 / libMesh::pi) * _local_pore_size * _local_Knudsen_number /
                         (1.0 + 2.0 * _local_Knudsen_number);
  auto tau_q = 0.5 + (3.0 + libMesh::pi * _A2 * torch::pow((2.0 * tau_s - 1.0), 2.0)) /
                         (8.0 * (2.0 * tau_s - 1.0));
  auto tau_d =
      0.5 + 3.0 * std::sqrt(3.0) / 8.0 * (_mfp / (_dx * (1.0 + 2.0 * _local_Knudsen_number)));

  auto ones = torch::ones_like(tau_s);

  _u.index_put_({Slice(), Slice(), Slice(), 0}, ones);
  _u.index_put_({Slice(), Slice(), Slice(), 1}, ones / 1.1);
  _u.index_put_({Slice(), Slice(), Slice(), 2}, ones / 1.2);
  _u.index_put_({Slice(), Slice(), Slice(), 3}, ones / tau_d);
  _u.index_put_({Slice(), Slice(), Slice(), 4}, ones / tau_q);
  _u.index_put_({Slice(), Slice(), Slice(), 5}, ones / tau_d);
  _u.index_put_({Slice(), Slice(), Slice(), 6}, ones / tau_q);
  _u.index_put_({Slice(), Slice(), Slice(), 7}, ones / tau_s);
  _u.index_put_({Slice(), Slice(), Slice(), 8}, ones / tau_s);

  _u_owned = ownedView(_u);
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
