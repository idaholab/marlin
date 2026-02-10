/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "Conversion.h"
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
  const auto inputs = getParam<TensorInputBufferName, std::string>("marlin_inputs", "neml2_inputs");
  std::map<neml2::LabeledAxisAccessor, TensorInputBufferName> lookup_marlin_name;
  const auto model_inputs = _model.consumed_items();
  mooseInfo("Inputs: ", Moose::stringify(model_inputs));

  // current inputs
  for (const auto & [marlin_input_name, neml2_input_name] : inputs)
  {
    const auto neml2_input =
        neml2::LabeledAxisAccessor(NEML2Utils::parseVariableName(neml2_input_name));

    // populate reverse lookup map
    if (lookup_marlin_name.find(neml2_input) != lookup_marlin_name.end())
      mooseError("Repeated NEML2 input ", neml2_input_name);
    lookup_marlin_name[neml2_input] = marlin_input_name;

    // the user should only specify current neml2 axis
    if (!neml2_input.is_state() && !neml2_input.is_force())
      mooseError("Specify only current forces or states as inputs. Old forces and states are "
                 "automatically coupled when needed.");

    // add input if the model requires it
    if (model_inputs.count(neml2_input))
    {
      const auto * input_buffer = &getInputBufferByName<>(marlin_input_name);
      const auto type = _model.input_variable(neml2_input).type();
      _input_mapping.emplace_back(input_buffer, type, neml2_input);
    }
  }

  // old state inputs
  for (const auto & neml2_input : model_inputs)
    if (neml2_input.is_old_state() || neml2_input.is_old_force())
    {
      // check if we couple the current state
      auto it = lookup_marlin_name.find(neml2_input.current());
      if (it == lookup_marlin_name.end())
        mooseError("The model requires ",
                   neml2_input,
                   " but no tensor buffer is assigned to ",
                   neml2_input.current(),
                   ".");
      const auto & marlin_input_name = it->second;

      const auto * old_states = &getBufferOldByName<>(marlin_input_name, 1);
      // we also get the current state here just to step zero, when no old state exists!
      const auto * input_buffer = &getInputBufferByName<>(marlin_input_name);
      const auto type = _model.input_variable(neml2_input).type();
      _old_input_mapping.emplace_back(old_states, input_buffer, type, neml2_input);
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
  auto insert_tensor = [&in, this](const auto & tensor, auto type, const auto & label)
  {
    // convert tensors on the fly at runtime
    auto sizes = tensor.sizes();
    if (sizes.size() == _dim && type == neml2::TensorType::kScalar)
      in[label] = neml2::Scalar(tensor);
    else if (sizes.size() == _dim + 1 && type == neml2::TensorType::kVec)
      in[label] = neml2::Vec(tensor, _domain.getShape());
    else if (sizes.size() == _dim + 3 && type == neml2::TensorType::kR2)
      in[label] = neml2::R2(tensor, _domain.getShape());
    else
      mooseError("Unsupported/mismatching tensor dimension");
  };

  // insert current state
  for (const auto & [current_state, type, label] : _input_mapping)
    insert_tensor(*current_state, type, label);

  // insert old state
  for (const auto & [old_states, current_state, type, label] : _old_input_mapping)
    if (old_states->empty())
      insert_tensor(*current_state, type, label);
    else
      insert_tensor((*old_states)[0], type, label);

  auto out = _model->value(in);

  for (const auto & [label, tensor_ptr] : _output_mapping)
    *tensor_ptr = out.at(label);
#endif
}
