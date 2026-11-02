/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "NEML2TensorCompute.h"
#include "NEML2Utils.h"

#ifdef NEML2_ENABLED
#include "neml2/csrc/aoti/Exception.h"
#endif

registerMooseObject("MarlinApp", NEML2TensorCompute);

InputParameters
NEML2TensorCompute::validParams()
{
  InputParameters params = TensorOperatorBase::validParams();
  params.addClassDescription(NEML2Utils::docstring("Compute object wrapper for a NEML2 model"));

  params.addRequiredParam<DataFileName>(
      "neml2_input_file",
      NEML2Utils::docstring("Path to the NEML2 input file containing the NEML2 model(s)."));
  params.addParam<std::vector<std::string>>(
      "cli_args",
      {},
      "Additional command line arguments to use when parsing the NEML2 input file.");

  params.addRequiredParam<std::string>("neml2_model", "Model name");

  params.addParam<std::vector<TensorInputBufferName>>(
      "marlin_inputs", "Marlin buffer names that go into the NEML2 model.");
  params.addParam<std::vector<std::string>>(
      "neml2_inputs", "NEML2 variable names corresponding to the `marlin_inputs`.");

  params.addParam<std::vector<std::string>>("neml2_outputs", "NEML2 model outputs.");
  params.addParam<std::vector<TensorInputBufferName>>(
      "marlin_outputs", "Marlin buffer name corresponding to the `neml2_outputs`.");

  return params;
}

NEML2TensorCompute::NEML2TensorCompute(const InputParameters & params)
  : TensorOperatorBase(params)
#ifdef NEML2_ENABLED
    ,
    _model(nullptr)
#endif
{
  NEML2Utils::assertNEML2Enabled();

#ifdef NEML2_ENABLED
  const auto fname = getParam<DataFileName>("neml2_input_file");
  const auto model_name = getParam<std::string>("neml2_model");

  const auto cli_args = getParam<std::vector<std::string>>("cli_args");
  if (!cli_args.empty())
    mooseError("The 'cli_args' parameter is not supported with the NEML2 v3 API.");

  // Use the same device and floating-point precision as the rest of Marlin's
  // tensor buffers so that the NEML2 model, its inputs, and its outputs agree.
  const at::Device device = MooseTensor::floatTensorOptions().device();
  const at::ScalarType dtype = MooseTensor::floatTensorOptions().dtype().toScalarType();

  // RAII guard so the global default dtype is restored even if model loading
  // throws.
  struct DefaultDtypeGuard
  {
    caffe2::TypeMeta saved;
    DefaultDtypeGuard(at::ScalarType d) : saved(at::get_default_dtype())
    {
      at::set_default_dtype(c10::scalarTypeToTypeMeta(d));
    }
    ~DefaultDtypeGuard() { at::set_default_dtype(saved); }
    DefaultDtypeGuard(const DefaultDtypeGuard &) = delete;
    DefaultDtypeGuard & operator=(const DefaultDtypeGuard &) = delete;
  } dtype_guard(dtype);

  _model = std::make_unique<neml2::eager::Model>(
      std::filesystem::path(std::string(fname)), model_name, device, std::vector<std::string>{});

  // The eager loader does not convert parameter storage, so do it here.
  for (const auto & [name, param] : _model->named_parameters())
    _model->set_parameter(name, param.to(device, dtype));

  for (const auto & [marlin_input_name, neml2_input_name] :
       getParam<TensorInputBufferName, std::string>("marlin_inputs", "neml2_inputs"))
  {
    const auto * input_buffer = &getInputBufferByName<>(marlin_input_name);
    _input_mapping.emplace_back(input_buffer, neml2_input_name);
  }

  for (const auto & [neml2_output_name, marlin_output_name] :
       getParam<std::string, TensorInputBufferName>("neml2_outputs", "marlin_outputs"))
  {
    auto * output_buffer = &getOutputBufferByName<>(marlin_output_name);
    _output_mapping.emplace_back(neml2_output_name, output_buffer);
  }
#endif
}

void
NEML2TensorCompute::init()
{
  // NEML2 v3 validates the model at construction time.
}

void
NEML2TensorCompute::computeBuffer()
{
#ifdef NEML2_ENABLED
  std::map<std::string, at::Tensor> in;
  for (const auto & [tensor_ptr, label] : _input_mapping)
    in[label] = tensor_ptr->to(_model->device(), _model->dtype());

  auto out = _model->forward(in);

  for (const auto & [label, tensor_ptr] : _output_mapping)
  {
    if (out.find(label) == out.end())
      mooseError("NEML2 model did not produce output '", label, "'.");
    *tensor_ptr = out.at(label).to(tensor_ptr->device(), tensor_ptr->dtype().toScalarType());
  }
#endif
}