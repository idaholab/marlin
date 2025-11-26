/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TracedComputeSequence.h"
#include "TensorProblem.h"

#include <mutex>

#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/graph_fuser.h>

// Global mutex for tracing (PyTorch JIT tracing is not thread-safe)
static std::mutex s_trace_mutex;

TracedComputeSequence::TracedComputeSequence() : _finalized(false) {}

void
TracedComputeSequence::addCompute(std::shared_ptr<TensorOperatorBase> compute)
{
  if (_finalized)
    throw std::runtime_error("Cannot add computes to a finalized TracedComputeSequence");
  _computes.push_back(compute);
}

void
TracedComputeSequence::finalize()
{
  if (_finalized)
    return;

  // Collect all inputs and outputs from the computes
  std::set<std::string> all_inputs, all_outputs;
  for (const auto & compute : _computes)
  {
    const auto & ins = compute->getRequestedItems();
    const auto & outs = compute->getSuppliedItems();
    all_inputs.insert(ins.begin(), ins.end());
    all_outputs.insert(outs.begin(), outs.end());
  }

  // External inputs = inputs that are not produced internally
  std::set_difference(all_inputs.begin(),
                      all_inputs.end(),
                      all_outputs.begin(),
                      all_outputs.end(),
                      std::inserter(_input_buffer_names, _input_buffer_names.begin()));

  // All outputs are tracked
  _output_buffer_names = all_outputs;

  // Create ordered lists for consistent stack ordering
  _ordered_input_names.assign(_input_buffer_names.begin(), _input_buffer_names.end());
  _ordered_output_names.assign(_output_buffer_names.begin(), _output_buffer_names.end());

  _finalized = true;
}

void
TracedComputeSequence::execute(TensorProblem & problem, const torch::TensorOptions & options)
{
  if (!_finalized)
    throw std::runtime_error("TracedComputeSequence must be finalized before execution");

  if (_computes.empty())
    return;

  // Skip JIT if we're already in a tracing context
  if (torch::jit::tracer::isTracing())
  {
    for (auto & compute : _computes)
      compute->computeBuffer();
    return;
  }

  // Compute current schema from input tensors
  std::vector<const torch::Tensor *> input_tensors;
  for (const auto & name : _ordered_input_names)
    input_tensors.push_back(&problem.getRawBuffer(name));

  TraceSchema schema = TraceSchema::fromTensors(input_tensors, options);

  // Cache lookup
  auto it = _traced_executors.find(schema);

  if (it != _traced_executors.end())
  {
    // Cache hit - execute traced graph
    auto stack = collectInputStack(problem);
    it->second.run(stack);
    assignOutputStack(stack, problem);
  }
  else
  {
    // Cache miss - trace and cache
    trace(problem, schema, options);
    // Re-execute using the newly cached trace
    execute(problem, options);
  }
}

void
TracedComputeSequence::invalidateCache()
{
  _traced_executors.clear();
}

torch::jit::Stack
TracedComputeSequence::collectInputStack(TensorProblem & problem) const
{
  torch::jit::Stack stack;
  for (const auto & name : _ordered_input_names)
    stack.emplace_back(problem.getBuffer<torch::Tensor>(name));
  return stack;
}

void
TracedComputeSequence::assignOutputStack(torch::jit::Stack & stack, TensorProblem & problem) const
{
  // GraphExecutor returns multiple outputs directly in the stack
  if (stack.size() != _ordered_output_names.size())
    throw std::runtime_error("Output stack size mismatch in TracedComputeSequence: expected " +
                             std::to_string(_ordered_output_names.size()) + " but got " +
                             std::to_string(stack.size()));

  for (std::size_t i = 0; i < _ordered_output_names.size(); ++i)
  {
    auto & buffer = problem.getBuffer<torch::Tensor>(_ordered_output_names[i]);
    buffer = stack[i].toTensor();
  }
}

void
TracedComputeSequence::assignInputStack(const torch::jit::Stack & stack,
                                        TensorProblem & problem) const
{
  if (stack.size() != _ordered_input_names.size())
    throw std::runtime_error("Input stack size mismatch in TracedComputeSequence");

  for (std::size_t i = 0; i < _ordered_input_names.size(); ++i)
  {
    auto & buffer = problem.getBuffer<torch::Tensor>(_ordered_input_names[i]);
    buffer = stack[i].toTensor();
  }
}

torch::jit::Stack
TracedComputeSequence::collectOutputStack(TensorProblem & problem) const
{
  // Return multiple outputs as individual stack entries (GraphExecutor supports this)
  torch::jit::Stack stack;
  for (const auto & name : _ordered_output_names)
    stack.emplace_back(problem.getBuffer<torch::Tensor>(name));
  return stack;
}

std::string
TracedComputeSequence::lookupBufferName(const torch::Tensor & tensor,
                                        TensorProblem & problem) const
{
  // Try to find this tensor in our known buffers
  for (const auto & name : _ordered_input_names)
    if (problem.getRawBuffer(name).data_ptr() == tensor.data_ptr())
      return "input::" + name;

  for (const auto & name : _ordered_output_names)
    if (problem.getRawBuffer(name).data_ptr() == tensor.data_ptr())
      return "output::" + name;

  return "";
}

void
TracedComputeSequence::trace(TensorProblem & problem,
                             const TraceSchema & schema,
                             const torch::TensorOptions & /*options*/)
{
  // Lock for thread safety - PyTorch JIT tracing is not thread-safe
  std::lock_guard<std::mutex> lock(s_trace_mutex);

  // Lambda that wraps the forward execution
  auto forward_wrap = [this, &problem](torch::jit::Stack inputs) -> torch::jit::Stack
  {
    assignInputStack(inputs, problem);
    for (auto & compute : _computes)
      compute->computeBuffer();
    return collectOutputStack(problem);
  };

  // Variable name lookup for better debugging
  auto var_name_lookup = [this, &problem](const torch::Tensor & var) -> std::string
  { return lookupBufferName(var, problem); };

  std::shared_ptr<torch::jit::tracer::TracingState> trace_state;

  try
  {
    auto result = torch::jit::tracer::trace(collectInputStack(problem),
                                            forward_wrap,
                                            var_name_lookup,
                                            /*strict=*/false,
                                            /*force_outplace=*/false);
    trace_state = std::get<0>(result);
  }
  catch (const std::exception & e)
  {
    throw std::runtime_error(std::string("Failed to trace compute sequence: ") + e.what());
  }

  // Apply optimizations to the traced graph
  torch::jit::EliminateDeadCode(trace_state->graph);
  torch::jit::ConstantPropagation(trace_state->graph);
  torch::jit::EliminateCommonSubexpression(trace_state->graph);
  torch::jit::FuseGraph(trace_state->graph, /*strict=*/true);

  // Create GraphExecutor from traced graph (supports multiple outputs unlike GraphFunction)
  _traced_executors.emplace(schema, torch::jit::GraphExecutor(trace_state->graph, "marlin_trace"));
}
