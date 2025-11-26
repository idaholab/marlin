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
#include "neml2/models/map_types_fwd.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
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
    _model(
        [this]()
        {
          const auto fname = getParam<DataFileName>("neml2_input_file");
          const auto cli_args = getParam<std::vector<std::string>>("cli_args");
          const auto model_name = getParam<std::string>("neml2_model");
          auto factory = neml2::load_input(std::string(fname), neml2::utils::join(cli_args, " "));
          return NEML2Utils::getModel(*factory, model_name);
        }())
#endif
{
  NEML2Utils::assertNEML2Enabled();

#ifdef NEML2_ENABLED
  for (const auto & [marlin_input_name, neml2_input_name] :
       getParam<TensorInputBufferName, std::string>("marlin_inputs", "neml2_inputs"))
  {
    const auto * input_buffer = &getInputBufferByName<>(marlin_input_name);
    _input_mapping.emplace_back(
        input_buffer, neml2::LabeledAxisAccessor(NEML2Utils::parseVariableName(neml2_input_name)));
  }

  for (const auto & [neml2_output_name, marlin_output_name] :
       getParam<std::string, TensorInputBufferName>("neml2_outputs", "marlin_outputs"))
  {
    auto * output_buffer = &getOutputBufferByName<>(marlin_output_name);
    _output_mapping.emplace_back(
        neml2::LabeledAxisAccessor(NEML2Utils::parseVariableName(neml2_output_name)),
        output_buffer);
  }
#endif
}

void
NEML2TensorCompute::init()
{
#ifdef NEML2_ENABLED
  neml2::diagnose(*_model);
#endif
}

void
NEML2TensorCompute::computeBuffer()
{
#ifdef NEML2_ENABLED
  neml2::ValueMap in;
  for (const auto & [tensor_ptr, label] : _input_mapping)
  {
    // convert tensors on the fly at runtime
    auto sizes = tensor_ptr->sizes();
    if (sizes.size() == _dim)
      in[label] = neml2::Scalar(*tensor_ptr);
    else if (sizes.size() == _dim + 1)
      in[label] = neml2::Vec(*tensor_ptr, _domain.getShape());
    else if (sizes.size() == _dim + 3)
      in[label] = neml2::R2(*tensor_ptr, _domain.getShape());
    else
      mooseError("Unsupported tensor dimension");
  }

  auto out = _model->value(in);

  for (const auto & [label, tensor_ptr] : _output_mapping)
    *tensor_ptr = out.at(label);
#endif
}
