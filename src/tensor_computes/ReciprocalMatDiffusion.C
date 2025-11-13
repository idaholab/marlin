
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ReciprocalMatDiffusion.h"
#include "MarlinUtils.h"
#include <torch/torch.h>

registerMooseObject("MarlinApp", ReciprocalMatDiffusion);

InputParameters
ReciprocalMatDiffusion::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription(
      "Calculates the divergence of flux for a variable mobility in reciprocal space.");
  params.addRequiredParam<TensorInputBufferName>("chemical_potential",
                                                 "Chemical potential buffer name");
  params.addRequiredParam<TensorInputBufferName>("mobility", "Mobility buffer name");
  params.addParam<TensorInputBufferName>("psi", "Variable to impose Neuamnn BC.");
  params.addParam<bool>("always_update_psi", false, "Set to true if the BC changes .");
  return params;
}

ReciprocalMatDiffusion::ReciprocalMatDiffusion(const InputParameters & parameters)
  : TensorOperator(parameters),
    _chem_pot(getInputBuffer("chemical_potential")),
    _M(getInputBuffer("mobility")),
    _imag(torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions())),
    _psi(getInputBuffer("psi")),
    _update_psi(true),
    _always_update_psi(getParam<bool>("always_update_psi"))
{
}

void
ReciprocalMatDiffusion::computeBuffer()
{
  if (_update_psi || _always_update_psi)
  {
    _psi_thresh = _psi > 0.0;
    _grad_psi_x_by_psi =
        torch::where(_psi_thresh, _domain.ifft(_i * _domain.fft(_psi) * _imag) / _psi, 0.0);
    _grad_psi_y_by_psi =
        torch::where(_psi_thresh, _domain.ifft(_j * _domain.fft(_psi) * _imag) / _psi, 0.0);
    _grad_psi_z_by_psi =
        torch::where(_psi_thresh, _domain.ifft(_k * _domain.fft(_psi) * _imag) / _psi, 0.0);
    _update_psi = false;
  }

  auto psi_M = _M * _psi_thresh;
  auto J_x = psi_M * _domain.ifft(_i * _domain.fft(_chem_pot) * _imag);
  auto J_y = psi_M * _domain.ifft(_j * _domain.fft(_chem_pot) * _imag);
  auto J_z = psi_M * _domain.ifft(_k * _domain.fft(_chem_pot) * _imag);

  auto div_J_hat = _imag * (_i * _domain.fft(J_x) + _j * _domain.fft(J_y) + _k * _domain.fft(J_z));
  auto no_flux_hat =
      _domain.fft(_grad_psi_x_by_psi * J_x + _grad_psi_y_by_psi * J_y + _grad_psi_z_by_psi * J_z);

  _u = div_J_hat + no_flux_hat;
}
