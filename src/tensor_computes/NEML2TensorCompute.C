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

  std::vector<std::string> devices = {at::Device(getMooseApp().getLibtorchDevice()).str()};
  std::vector<std::size_t> batch_sizes = {0};
  const auto mpi_comm = comm().get();

  _model = makeNEML2ModelHandle(
      true, std::string(fname), model_name, devices, batch_sizes, &mpi_comm, {});

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
    in[label] = *tensor_ptr;

  auto out = _model->value(in);

  for (const auto & [label, tensor_ptr] : _output_mapping)
  {
    if (out.find(label) == out.end())
      mooseError("NEML2 model did not produce output '", label, "'.");
    *tensor_ptr = out.at(label);
  }
#endif
}
