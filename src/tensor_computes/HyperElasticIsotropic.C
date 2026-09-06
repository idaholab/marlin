/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MooseError.h"
#include "HyperElasticIsotropic.h"

registerMooseObject("MarlinApp", HyperElasticIsotropic);

InputParameters
HyperElasticIsotropic::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Hyperelastic isotropic constitutive model.");
  params.addRequiredParam<TensorInputBufferName>("F", "Deformation gradient tensor");
  params.addParam<TensorInputBufferName>("Fstar", "Optional eigenstrain tensor");
  params.addRequiredParam<TensorInputBufferName>("mu", "Deformation gradient tensor");
  params.addRequiredParam<TensorInputBufferName>("K", "Deformation gradient tensor");
  params.addParam<TensorOutputBufferName>("tangent_operator", "dstressdstrain", "Stiffness tensor");
  return params;
}

HyperElasticIsotropic::HyperElasticIsotropic(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _ti(torch::eye(_dim, MooseTensor::floatTensorOptions())),
    _tI(MooseTensor::unsqueeze0(_ti, _dim)),
    _tI4(MooseTensor::unsqueeze0(torch::einsum("il,jk", {_ti, _ti}), _dim)),
    _tI4rt(MooseTensor::unsqueeze0(torch::einsum("ik,jl", {_ti, _ti}), _dim)),
    _tI4s((_tI4 + _tI4rt) / 2.0),
    _tII(MooseTensor::dyad22(_tI, _tI)),
    _tF(getInputBuffer("F")),
    _pFstar(isParamValid("Fstar") ? &getInputBuffer("Fstar") : nullptr),
    _tmu(getInputBuffer("mu")),
    _tK(getInputBuffer("K")),
    _tK4(getOutputBuffer("tangent_operator"))
{
}

void
HyperElasticIsotropic::computeBuffer()
{
  using namespace MooseTensor;

  const auto F_e = _pFstar ? dot22(_tF, torch::inverse(*_pFstar)) : _tF;

  // Compute Green–Lagrange strain E = 0.5 (F_e^T * F_e - I)
  const auto E = 0.5 * (dot22(trans2(F_e), F_e) - _tI);

  // Build material stiffness tensor
  const auto C4 = _tK.reshape(_domain.getValueShape({1, 1, 1, 1})) * _tII +
                  2. * _tmu.reshape(_domain.getValueShape({1, 1, 1, 1})) * (_tI4s - 1. / 3. * _tII);

  // Second Piola-Kirchhoff stress: S = C : E
  const auto S = ddot42(C4, E);

  // First Piola-Kirchhoff stress: P = F_e * S
  _u = dot22(F_e, S);

  _tK4 = dot24(S, _tI4) + ddot44(ddot44(_tI4rt, dot42(dot24(F_e, C4), trans2(F_e))), _tI4rt);

  // Consistent tangent: K_ijkl = F^e_{im} C_{jlmn} F^e_{kn} + delta_{ik} S_{jl}
  // const auto K_geo = torch::einsum("...im,...jlmn,...kn->...ijkl", {F_e, C4, F_e});
  // const auto K_stress = torch::einsum("ik,jl,...jl->...ijkl", {_ti, _ti, S});
  // _tK4 = K_geo + K_stress;
}
