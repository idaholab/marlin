/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMStackTensors.h"

registerMooseObject("MarlinApp", LBMStackTensors);

InputParameters
LBMStackTensors::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();

  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "inputs", "Names of input tensor buffers to stack.");
  params.addClassDescription("Stack given scalar tensor buffers and output vectorial tensor.");

  return params;
}

LBMStackTensors::LBMStackTensors(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _buffer_names(getParam<std::vector<TensorInputBufferName>>("inputs"))
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
LBMStackTensors::init()
{
  // make sure output buffer has the same dimensions
  if (_u.dim() < 4)
    mooseError("Output buffer must be vectorial tensor.");
}

void
LBMStackTensors::computeBuffer()
{
  using torch::indexing::Slice;

  std::vector<torch::Tensor> tensor_vector;
  for (const auto & name : _buffer_names)
  {
    auto tensor_buffer = getInputBufferByName(name, _radius);

    if (tensor_buffer.dim() < 3)
      tensor_buffer = tensor_buffer.unsqueeze(2);
    if (tensor_buffer.dim() > 3)
    {
      std::string error_msg = "Input buffer ";
      error_msg.append(name);
      error_msg += " must be scalar";
      mooseError(error_msg);
    }
    tensor_vector.push_back(tensor_buffer);
  }

  // Stack the tensors along a new dimension
  _u = torch::stack(tensor_vector, 3);
}
