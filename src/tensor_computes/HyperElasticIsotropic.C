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
    _tmu(getInputBuffer("mu")),
    _tK(getInputBuffer("K")),
    _tK4(getOutputBuffer("tangent_operator"))
{
}

void
HyperElasticIsotropic::computeBuffer()
{
  using namespace MooseTensor;

  const auto C4 = _tK.reshape(_domain.getValueShape({1, 1, 1, 1})) * _tII +
                  2. * _tmu.reshape(_domain.getValueShape({1, 1, 1, 1})) * (_tI4s - 1. / 3. * _tII);
  const auto S = ddot42(C4, .5 * (dot22(trans2(_tF), _tF) - _tI));

  _u = dot22(_tF, S);
  _tK4 = dot24(S, _tI4) + ddot44(ddot44(_tI4rt, dot42(dot24(_tF, C4), trans2(_tF))), _tI4rt);
}
