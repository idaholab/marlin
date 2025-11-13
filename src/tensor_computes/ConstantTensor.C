/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ConstantTensor.h"
#include "MarlinUtils.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", ConstantTensor);
registerMooseObject("MarlinApp", ConstantReciprocalTensor);

template <bool reciprocal>
InputParameters
ConstantTensorTempl<reciprocal>::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addParam<MarlinConstantName>("imaginary", "0.0", "Imaginary part of the constant value.");
  if constexpr (reciprocal)
    params.addClassDescription("Constant tensor in reciprocal space.");
  else
  {
    params.addClassDescription("Constant tensor in real space.");
    params.suppressParameter<MarlinConstantName>("imaginary");
  }
  params.addParam<MarlinConstantName>("real", "0.0", "Real part of the constant value.");
  params.addParam<bool>("full", false, "Construct a full tensor will all entries");
  return params;
}

template <bool reciprocal>
ConstantTensorTempl<reciprocal>::ConstantTensorTempl(const InputParameters & parameters)
  : TensorOperator(parameters),
    _dim(_domain.getDim()),
    _real(this->getConstant<Real>("real")),
    _imaginary(this->getConstant<Real>("imaginary"))
{
}

template <bool reciprocal>
void
ConstantTensorTempl<reciprocal>::computeBuffer()
{
  if constexpr (reciprocal)
    _u = torch::complex(
        torch::full(_domain.getReciprocalShape(), _real, MooseTensor::floatTensorOptions()),
        torch::full(_domain.getReciprocalShape(), _imaginary, MooseTensor::floatTensorOptions()));
  else
    _u = torch::full(_domain.getShape(), _real, MooseTensor::floatTensorOptions());
}

template class ConstantTensorTempl<true>;
template class ConstantTensorTempl<false>;
