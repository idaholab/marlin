/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "StackTensors.h"

#include <set>

registerMooseObject("MarlinApp", StackTensors);

InputParameters
StackTensors::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "inputs", "Names of input tensor buffers to stack.");
  params.addParam<int>("stack_dim", -1, "Dimension to stack along (default: last).");
  params.addClassDescription("Stack given scalar tensor buffers into a vector tensor.");
  return params;
}

StackTensors::StackTensors(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _buffer_names(getParam<std::vector<TensorInputBufferName>>("inputs")),
    _stack_dim(getParam<int>("stack_dim"))
{
  // check for duplicates
  auto hasDuplicates = [](const std::vector<std::string> & values)
  {
    std::set<std::string> s(values.begin(), values.end());
    return values.size() != s.size();
  };

  if (hasDuplicates(_buffer_names))
    paramError("inputs", "Duplicate buffer name.");
}

void
StackTensors::computeBuffer()
{
  std::vector<torch::Tensor> tensor_vector;
  tensor_vector.reserve(_buffer_names.size());

  for (const auto & name : _buffer_names)
  {
    auto tensor_buffer = getInputBufferByName(name);
    tensor_vector.push_back(tensor_buffer);
  }

  _u = torch::stack(tensor_vector, _stack_dim);
}
