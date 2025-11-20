
/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ReciprocalAllenCahn.h"
#include "MarlinUtils.h"
#include <torch/torch.h>

registerMooseObject("MarlinApp", ReciprocalAllenCahn);

InputParameters
ReciprocalAllenCahn::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("Calculates the Allen-Cahn bulk driving force masked using psi.");
  params.addRequiredParam<TensorInputBufferName>("dF_chem_deta", "Driving force buffer name");
  params.addRequiredParam<TensorInputBufferName>("L", "Allen-Cahn mobility buffer name");
  params.addRequiredParam<TensorInputBufferName>("psi", "Variable to impose Neumann BC.");
  params.addParam<bool>("always_update_psi", false, "Set to true if the BC changes .");
  return params;
}

ReciprocalAllenCahn::ReciprocalAllenCahn(const InputParameters & parameters)
  : TensorOperator(parameters),
    _dF_chem_deta(getInputBuffer("dF_chem_deta")),
    _L(getInputBuffer("L")),
    _imag(torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions())),
    _psi(getInputBuffer("psi")),
    _update_psi(true),
    _always_update_psi(getParam<bool>("always_update_psi"))
{
}

void
ReciprocalAllenCahn::computeBuffer()
{
  if (_update_psi || _always_update_psi)
  {
    _psi_thresh = _psi > 0.0;
    _update_psi = false;
  }

  auto rate = torch::where(_psi_thresh, -1 * _L * _dF_chem_deta, 0.0);
  _u = _domain.fft(rate);
}
