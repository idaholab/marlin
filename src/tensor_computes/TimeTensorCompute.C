/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SwiftUtils.h"
#include "TimeTensorCompute.h"

registerMooseObject("SwiftApp", TimeTensorCompute);

InputParameters
TimeTensorCompute::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Scalar tensor with the current simulation time.");
  return params;
}

TimeTensorCompute::TimeTensorCompute(const InputParameters & parameters)
  : TensorOperator<>(parameters)
{
}

void
TimeTensorCompute::computeBuffer()
{
  _u = torch::tensor({_time}, MooseTensor::floatTensorOptions());
}
