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

  // Place the NEML2 model on the same device as Marlin's tensor buffers. The
  // eager model's dtype is taken from its first floating-point parameter (NEML2
  // builds those in double precision) and falls back to the torch default dtype
  // for parameter-free models, so set that default to Marlin's precision while
  // loading. Remaining mismatches are reconciled in computeBuffer().
  // MPS has no float64 support, so the model stays on the CPU in that case.
  at::Device device = MooseTensor::floatTensorOptions().device();
  if (device.is_mps())
    device = at::Device(at::kCPU);

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
  };

  const auto load_model = [&]()
  {
    DefaultDtypeGuard dtype_guard(dtype);
    return std::make_unique<neml2::eager::Model>(
        std::filesystem::path(std::string(fname)), model_name, device);
  };

  _model = load_model();
  // The first eager model in the process imports torch into the embedded Python
  // interpreter, which resets the global default dtype to float32 mid-load. For a
  // parameter-free model (whose dtype comes from that default) load once more;
  // torch is initialized by now, so the guard takes effect. Models with
  // parameters are always float64 and are not reloaded.
  if (_model->dtype() != dtype && _model->parameter_base_shapes().empty())
    _model = load_model();

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
  // Change device and dtype in separate steps so that a float64 conversion never
  // runs on a device without float64 support (MPS).
  std::map<std::string, at::Tensor> in;
  for (const auto & [tensor_ptr, label] : _input_mapping)
    in[label] = tensor_ptr->to(_model->device()).to(_model->dtype());

  auto out = _model->forward(in);

  for (const auto & [label, tensor_ptr] : _output_mapping)
  {
    if (out.find(label) == out.end())
      mooseError("NEML2 model did not produce output '", label, "'.");
    *tensor_ptr = out.at(label)
                      .to(MooseTensor::floatTensorOptions().dtype().toScalarType())
                      .to(MooseTensor::floatTensorOptions().device());
  }
#endif
}
