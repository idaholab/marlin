/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LibtorchGibbsEnergy.h"
#include "SwiftUtils.h"
#include <valarray>
#include <vector>

registerMooseObject("SwiftApp", LibtorchGibbsEnergy);

InputParameters
LibtorchGibbsEnergy::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Calculates Gibbs energy, chemical potential, and driving force for "
                             "order parameters using a Libtorch model.");
  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "phase_fractions", "Phase fractions in order of inputs to torch model.");
  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "concentrations", "Concentrations in order of inputs to torch model.");
  params.addRequiredParam<std::vector<TensorInputBufferName>>("domega_detas",
                                                              "Buffer names for AC driving force.");
  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "chem_pots", "Buffer names for chemical potentials.");

  params.addRequiredParam<DataFileName>(
      "libtorch_model_file", "Path to the Libtorch file containing the Gibbs energy model.");
  return params;
}

LibtorchGibbsEnergy::LibtorchGibbsEnergy(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _file_path(Moose::DataFileUtils::getPath(getParam<DataFileName>("libtorch_model_file"))),
    _surrogate(std::make_unique<torch::jit::script::Module>(torch::jit::load(_file_path.path)))
{
  _surrogate->to(MooseTensor::floatTensorOptions().device());
  _surrogate->eval();

  auto phase_fractions = getParam<std::vector<TensorInputBufferName>>("phase_fractions");
  auto domega_detas = getParam<std::vector<TensorOutputBufferName>>("domega_detas");

  _n_phases = phase_fractions.size();
  if (_n_phases != domega_detas.size())
    mooseError("Number of phases must match number of domega_deta buffers.");

  for (const auto & name : phase_fractions)
    _phase_fractions.push_back(&getInputBufferByName(name));
  for (const auto & name : domega_detas)
    _domega_detas.push_back(&getOutputBufferByName(name));

  auto concentrations = getParam<std::vector<TensorInputBufferName>>("concentrations");
  auto chemical_potentials = getParam<std::vector<TensorOutputBufferName>>("chem_pots");
  _n_components = concentrations.size();
  if (_n_components != chemical_potentials.size())
    mooseError("Number of concentrations must match number of chemical potential buffers.");

  for (const auto & name : concentrations)
    _concentrations.push_back(&getInputBufferByName(name));
  for (const auto & name : chemical_potentials)
    _chemical_potentials.push_back(&getOutputBufferByName(name));
}

void
LibtorchGibbsEnergy::computeBuffer()
{
  std::vector<torch::Tensor> X_vec;
  for (const auto * tensor_ptr : _phase_fractions)
    X_vec.push_back(*tensor_ptr);
  for (const auto * tensor_ptr : _concentrations)
    X_vec.push_back(*tensor_ptr);

  // Stack the tensors along the last axis (-1)
  torch::Tensor X = torch::stack(X_vec, -1);

  // flatten
  auto input_size = X.size(-1);
  auto batch_dims = X.sizes().slice(0, X.dim() - 1);
  auto batch_size = X.numel() / input_size;

  auto X_flat = X.reshape({batch_size, input_size}).contiguous();
  X_flat.set_requires_grad(true);

  auto G = _surrogate->forward({X_flat}).toTensor().squeeze();

  std::vector<torch::Tensor> grads = torch::autograd::grad({G.sum()},
                                                           {X_flat},
                                                           /*grad_outputs=*/{},
                                                           /*retain_graph=*/false,
                                                           /*create_graph=*/false,
                                                           /*allow_unused=*/false);
  auto jacobian_flat = grads[0];

  std::vector<int64_t> output_shape(batch_dims.begin(), batch_dims.end());

  for (const auto i : make_range(_n_phases))
    *_domega_detas[i] = jacobian_flat
                            .slice(
                                /*dim=*/1,
                                /*start=*/i,
                                /*end=*/i + 1)
                            .reshape(output_shape);

  for (const auto i : make_range(_n_components))
    *_chemical_potentials[i] = jacobian_flat
                                   .slice(
                                       /*dim=*/1,
                                       /*start=*/_n_phases + i,
                                       /*end=*/_n_phases + i + 1)
                                   .reshape(output_shape);
  _u = G.reshape(output_shape);
}
