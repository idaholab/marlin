/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ComputeDisplacements.h"
#include "MooseError.h"
#include "DomainAction.h"
#include "MarlinUtils.h"
#include <ATen/core/TensorBody.h>

registerMooseObject("MarlinApp", ComputeDisplacements);

InputParameters
ComputeDisplacements::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Compute updated displacements from the deformation gradient tensor.");
  params.addRequiredParam<TensorInputBufferName>("F", "Deformation gradient tensor.");
  return params;
}

ComputeDisplacements::ComputeDisplacements(const InputParameters & parameters)
  : TensorOperator<>(parameters), _deformation_gradient_tensor(getInputBuffer("F"))
{
}

void
saveDebug(const torch::Tensor & debug)
{
  // dump Ghat
  MooseTensor::printTensorInfo("debug", debug);

  std::size_t raw_size = debug.numel();
  char * raw_ptr = static_cast<char *>(debug.data_ptr());

  if (debug.dtype() == torch::kFloat32)
    raw_size *= 4;
  else if (debug.dtype() == torch::kFloat64)
    raw_size *= 8;
  else
    mooseError("Unsupported output type");

  auto file = std::fstream("debug.bin", std::ios::out | std::ios::binary);
  file.write(raw_ptr, raw_size);
  file.close();
}

void
ComputeDisplacements::computeBuffer()
{
  const auto & F = _deformation_gradient_tensor;
  if (!F.defined())
    return;

  // compute strain gradient tensor H
  mooseAssert(
      F.size(-1) == _dim && F.size(-2) == _dim,
      "Value dimensions of the deformation gradient tensor to not match the problem dimension");

  const auto I3 = torch::eye(_dim, F.options());

  const auto Fbox = _domain.average(F);

  // const auto Hbar = _domain.fft(F - MooseTensor::unsqueeze0(Fbox, _dim));
  const auto Hbar = _domain.fft(F - Fbox);

  const auto q = _domain.getKGrid() * (-_imaginary);
  const auto Q = _domain.getKSquare();

  const auto numer = torch::einsum("...ij,...j->...i", {Hbar, q});
  const auto denom = Q.unsqueeze(-1);

  const auto u_periodic_bar = torch::where(denom == 0, 0.0, numer / denom);

  torch::Tensor u_periodic;
  torch::Tensor u_aff;

  const auto & X = _domain.getXGrid();
  u_aff = torch::einsum("ij,...j->...i", {Fbox - I3, X});
  u_periodic = _domain.ifft(u_periodic_bar);

  std::vector<int64_t> shape(_domain.getShape().begin(), _domain.getShape().end());
  for (auto & n : shape)
    n++;

  namespace tf = torch::nn::functional;
  auto interpolate = [&](auto mode)
  {
    _u = tf::interpolate((u_aff + u_periodic).movedim(-1, 0).unsqueeze(1),
                         tf::InterpolateFuncOptions().size(shape).mode(mode).align_corners(true))
             .squeeze(1)
             .movedim(0, -1);
  };

  if (_dim == 3)
    interpolate(torch::kTrilinear);
  else if (_dim == 2)
    interpolate(torch::kBilinear);
  else if (_dim == 1)
    interpolate(torch::kLinear);
  else
    mooseError("Unsupported problem dimension");
}
