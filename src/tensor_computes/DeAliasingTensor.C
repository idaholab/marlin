/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "DeAliasingTensor.h"
#include "MooseEnum.h"
#include "MarlinUtils.h"
#include "wasplsp/LSP.h"

registerMooseObject("MarlinApp", DeAliasingTensor);

InputParameters
DeAliasingTensor::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Create a de-aliasing filter.");
  MooseEnum deAliasingMethod("SHARP HOULI");
  params.addRequiredParam<MooseEnum>("method", deAliasingMethod, "Prefactor");
  params.addParam<Real>("p", 16, "Hou-Li filter exponent");
  params.addParam<Real>("alpha", 36, "Hou-Li filter pre-factor");

  return params;
}

DeAliasingTensor::DeAliasingTensor(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _method(getParam<MooseEnum>("method").getEnum<DeAliasingMethod>()),
    _p(getParam<Real>("p")),
    _alpha(getParam<Real>("alpha"))
{
}

void
DeAliasingTensor::computeBuffer()
{
  // maximum frequency in each direction
  auto imax = torch::max(torch::abs(_i)).item<double>();
  auto jmax = torch::max(torch::abs(_j)).item<double>();
  auto kmax = torch::max(torch::abs(_k)).item<double>();

  switch (_method)
  {
    case DeAliasingMethod::SHARP:
      _u = torch::where((torch::abs(_i) > 2 * imax / 3) | (torch::abs(_j) > 2 * jmax / 3) |
                            (torch::abs(_k) > 2 * kmax / 3),
                        0.0,
                        1.0);
      break;

    case DeAliasingMethod::HOULI:
      auto px = torch::pow(torch::abs(_i) / (imax ? imax : 1.0), _p);
      auto py = torch::pow(torch::abs(_j) / (jmax ? jmax : 1.0), _p);
      auto pz = torch::pow(torch::abs(_k) / (kmax ? kmax : 1.0), _p);

      _u = torch::exp(-_alpha * (px + py + pz));
      break;
  }
}
