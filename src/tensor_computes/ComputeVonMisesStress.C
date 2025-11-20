/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ComputeVonMisesStress.h"
#include "MooseError.h"
#include "DomainAction.h"
#include "MarlinUtils.h"
#include <ATen/core/TensorBody.h>

registerMooseObject("MarlinApp", ComputeVonMisesStress);

InputParameters
ComputeVonMisesStress::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Compute vonMises stress.");
  params.addParam<TensorInputBufferName>("stress", "stress", "Stress tensor.");
  return params;
}

ComputeVonMisesStress::ComputeVonMisesStress(const InputParameters & parameters)
  : TensorOperator<>(parameters), _stress(getInputBuffer("stress"))
{
}

void
ComputeVonMisesStress::computeBuffer()
{
  using namespace torch::indexing;
  if (!_stress.defined())
    return;

  if (_dim == 3)
  {
    auto stress_xx = _stress.index({Ellipsis, 0, 0});
    auto stress_yy = _stress.index({Ellipsis, 1, 1});
    auto stress_zz = _stress.index({Ellipsis, 2, 2});
    auto stress_xy = _stress.index({Ellipsis, 0, 1});
    auto stress_yz = _stress.index({Ellipsis, 1, 2});
    auto stress_zx = _stress.index({Ellipsis, 2, 0});

    auto term1 = (stress_xx - stress_yy).pow(2);
    auto term2 = (stress_yy - stress_zz).pow(2);
    auto term3 = (stress_zz - stress_xx).pow(2);
    auto term4 = 6 * (stress_xy.pow(2) + stress_yz.pow(2) + stress_zx.pow(2));

    _u = torch::sqrt(0.5 * (term1 + term2 + term3 + term4));
  }
  else if (_dim == 2)
  {
    auto stress_xx = _stress.index({Ellipsis, 0, 0});
    auto stress_yy = _stress.index({Ellipsis, 1, 1});
    auto stress_xy = _stress.index({Ellipsis, 0, 1});

    auto term1 = (stress_xx - stress_yy).pow(2);
    auto term2 = 6 * stress_xy.pow(2);

    _u = torch::sqrt(0.5 * (term1 + term2));
  }
  else
    mooseError("Unsupported problem dimension ", _dim);
}
