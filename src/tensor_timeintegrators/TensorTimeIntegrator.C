/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorTimeIntegrator.h"
#include "TensorProblem.h"

template <typename T>
InputParameters
TensorTimeIntegrator<T>::validParams()
{
  InputParameters params = TensorOperator<T>::validParams();
  params.registerBase("TensorTimeIntegrator");
  params.addClassDescription("TensorTimeIntegrator object.");
  return params;
}

template <typename T>
TensorTimeIntegrator<T>::TensorTimeIntegrator(const InputParameters & parameters)
  : TensorOperator<T>(parameters), _sub_dt(this->_tensor_problem.subDt())
{
}

template <typename T>
const std::vector<T> &
TensorTimeIntegrator<T>::getBufferOld(const std::string & param, unsigned int max_states)
{
  return getBufferOldByName(this->template getParam<TensorInputBufferName>(param), max_states);
}

template <typename T>
const std::vector<T> &
TensorTimeIntegrator<T>::getBufferOldByName(const TensorInputBufferName & buffer_name,
                                            unsigned int max_states)
{
  return this->_tensor_problem.template getBufferOld<T>(buffer_name, max_states);
}

template class TensorTimeIntegrator<torch::Tensor>;
