/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorOperatorBase.h"
#include "TensorBuffer.h"
#include "DomainAction.h"

InputParameters
TensorOperatorBase::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.registerBase("TensorOperator");
  params.addPrivateParam<TensorProblem *>("_tensor_problem", nullptr);
  params.addPrivateParam<const DomainAction *>("_domain", nullptr);
  params.addClassDescription("TensorOperatorBase object.");
  return params;
}

TensorOperatorBase::TensorOperatorBase(const InputParameters & parameters)
  : MooseObject(parameters),
    DependencyResolverInterface(),
    MarlinConstantInterface(parameters),
    _requested_buffers(),
    _supplied_buffers(),
    _tensor_problem(*getCheckedPointerParam<TensorProblem *>("_tensor_problem")),
    _domain(*getCheckedPointerParam<const DomainAction *>("_domain")),
    _x(_domain.getAxis(0)),
    _y(_domain.getAxis(1)),
    _z(_domain.getAxis(2)),
    _i(_domain.getReciprocalAxis(0)),
    _j(_domain.getReciprocalAxis(1)),
    _k(_domain.getReciprocalAxis(2)),
    _imaginary(
        torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions())),
    _time(_tensor_problem.subTime()),
    _dim(_domain.getDim())
{
}

TensorOperatorBase &
TensorOperatorBase::getCompute(const std::string & param_name)
{
  const auto name = getParam<TensorComputeName>(param_name);
  for (const auto & cmp : _tensor_problem.getComputes())
    if (cmp->name() == name)
      return *cmp;
  paramError(param_name, "Compute not found.");
}

bool
TensorOperatorBase::usesParallelFFT() const
{
  return _domain.isParallelFFT();
}
