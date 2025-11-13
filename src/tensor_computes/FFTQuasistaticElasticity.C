
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FFTQuasistaticElasticity.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", FFTQuasistaticElasticity);

InputParameters
FFTQuasistaticElasticity::validParams()
{
  InputParameters params = TensorOperatorBase::validParams();
  params.addClassDescription("FFT based monolithic homogeneous quasistatic elasticity solve.");
  params.addParam<std::vector<TensorOutputBufferName>>("displacements", "Displacements");
  params.addParam<TensorInputBufferName>("cbar", "FFT of concentration buffer");
  params.addRequiredParam<Real>("mu", "Lame mu");
  params.addRequiredParam<Real>("lambda", "Lame lambda");
  params.addRequiredParam<Real>("e0", "volumetric eigenstrain");
  return params;
}

FFTQuasistaticElasticity::FFTQuasistaticElasticity(const InputParameters & parameters)
  : TensorOperatorBase(parameters),
    _two_pi_i(torch::tensor(c10::complex<double>(0.0, 2.0 * pi),
                            MooseTensor::complexFloatTensorOptions())),
    _mu(getParam<Real>("mu")),
    _lambda(getParam<Real>("lambda")),
    _e0(getParam<Real>("e0")),
    _cbar(getInputBuffer("cbar"))
{
  for (const auto & name : getParam<std::vector<TensorOutputBufferName>>("displacements"))
    _displacements.push_back(&getOutputBufferByName(name));

  if (_domain.getDim() != _displacements.size())
    paramError("displacements", "Need one displacement variable per mesh dimension");
}

void
FFTQuasistaticElasticity::computeBuffer()
{
  // const auto & ux = *_displacements[0];
  // const auto & uy = *_displacements[1];
  // const auto & uz = *_displacements[2];

  // // FFT displacements
  // auto uxbar = _domain.fft(ux);
  // auto uybar = _domain.fft(uy);
  // auto uzbar = _domain.fft(uz);

  // strain tensor (in reciprocal space)
  // const auto exx = uxbar * _two_pi_i * _i;
  // const auto eyy = uybar * _two_pi_i * _j;
  // const auto ezz = uzbar * _two_pi_i * _k;
  // const auto exy = 0.5 * (uxbar * _two_pi_i * _j + uybar * _two_pi_i * _i);
  // const auto exz = 0.5 * (uxbar * _two_pi_i * _k + uzbar * _two_pi_i * _i);
  // const auto eyz = 0.5 * (uybar * _two_pi_i * _k + uzbar * _two_pi_i * _j);

  // precalculate these!
  const auto ul = 2.0 * _mu + _lambda;
  const auto kx = _two_pi_i * _i;
  const auto ky = _two_pi_i * _j;
  const auto kz = _two_pi_i * _k;

  // system matrix ()
  const auto Axx = ul * kx * kx + _mu * ky * ky + _mu * kz * kz;
  const auto s = Axx.sizes();
  const auto Axy = ((_lambda + _mu) * kx * ky).expand(s);
  const auto Axz = ((_lambda + _mu) * kx * kz).expand(s);
  const auto Ayy = ul * ky * ky + _mu * kx * kx + _mu * kz * kz;
  const auto Ayz = ((_lambda + _mu) * ky * kz).expand(s);
  const auto Azz = ul * kz * kz + _mu * kx * kx + _mu * ky * ky;

  // override Axx, Ayy, Azz for |k|=0
  Axx.index({0, 0, 0}) = 1.0;
  Ayy.index({0, 0, 0}) = 1.0;
  Azz.index({0, 0, 0}) = 1.0;

  // RHS (eigenstrain)
  using torch::stack;
  const auto e = 2.0 * _e0 * _cbar * (3.0 * _lambda + _mu);
  e.index({0, 0, 0}) = 0.0;

  const auto b = stack({kx * e, ky * e, kz * e}, -1);

  const auto A = stack(
      {stack({Axx, Axy, Axz}, -1), stack({Axy, Ayy, Ayz}, -1), stack({Axz, Ayz, Azz}, -1)}, -1);

  // solve
  const auto x = at::linalg_solve(A, b, true);

  // inverse transform the solution
  using torch::indexing::Slice;
  for (const auto i : make_range(3))
  {
    const auto slice = torch::squeeze(x.index({Slice(), Slice(), Slice(), i}), -1);
    *_displacements[i] = _domain.ifft(slice);
  }
}
