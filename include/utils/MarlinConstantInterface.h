/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MarlinTypes.h"
#include "TensorProblem.h"

class InputParameters;

class MarlinConstantInterface
{
public:
  MarlinConstantInterface(const InputParameters & params);

  template <typename T>
  const T & getConstant(const std::string & param_name);
  template <typename T>
  const T & getConstantByName(const MarlinConstantName & name);

  template <typename T>
  void declareConstant(const std::string & param_name, const T & value);
  template <typename T>
  void declareConstantByName(const MarlinConstantName & name, const T & value);

protected:
  const InputParameters & _params;
  TensorProblem & _sci_tensor_problem;
};

template <typename T>
const T &
MarlinConstantInterface::getConstant(const std::string & param_name)
{
  return getConstantByName<T>(_params.get<MarlinConstantName>(param_name));
}

template <typename T>
const T &
MarlinConstantInterface::getConstantByName(const MarlinConstantName & name)
{
  return _sci_tensor_problem.getConstant<T>(name);
}

template <typename T>
void
MarlinConstantInterface::declareConstant(const std::string & param_name, const T & value)
{
  declareConstantByName<T>(_params.get<MarlinConstantName>(param_name), value);
}

template <typename T>
void
MarlinConstantInterface::declareConstantByName(const MarlinConstantName & name, const T & value)
{
  _sci_tensor_problem.declareConstant<T>(name, value);
}
