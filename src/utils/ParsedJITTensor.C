/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ParsedJITTensor.h"
#include "SwiftUtils.h"

#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/autograd/grad_mode.h>

ParsedJITTensor::ParsedJITTensor() {}

bool
ParsedJITTensor::parse(const std::string & expression,
                       const std::vector<std::string> & variables,
                       const std::unordered_map<std::string, torch::Tensor> & constants)
{
  _variables = variables;
  _constants = constants;
  _error.clear();
  _graph.reset();
  _executor.reset();

  // Extract constant names for the parser
  std::unordered_set<std::string> constant_names;
  for (const auto & kv : constants)
    constant_names.insert(kv.first);

  _ast = _parser.parse(expression, constant_names);
  if (!_ast)
  {
    _error = _parser.errorMessage();
    return false;
  }

  return true;
}

void
ParsedJITTensor::differentiate(const std::string & var)
{
  if (!_ast)
  {
    _error = "No expression to differentiate";
    return;
  }

  _ast = _ast->differentiate(var);
  _graph.reset();
  _executor.reset();
}

void
ParsedJITTensor::compile()
{
  if (!_ast)
  {
    _error = "No expression to compile";
    return;
  }

  // Optimize the AST first
  _ast = _ast->simplify();

  // Build a JIT graph
  _graph = std::make_shared<torch::jit::Graph>();
  std::unordered_map<std::string, torch::jit::Value *> var_map;

  // Create input nodes for variables
  for (const auto & var : _variables)
  {
    auto input = _graph->addInput();
    var_map[var] = input;
  }

  // Add constants as graph inputs
  for (const auto & kv : _constants)
  {
    auto input = _graph->addInput();
    var_map[kv.first] = input;
  }

  // Build the graph from AST
  auto output = _ast->buildGraph(*_graph, var_map);

  // Register the single output value
  _graph->registerOutput(output);

  // Lint and optimize graph
  _graph->lint();

  // Apply torch optimizations
  torch::jit::EliminateDeadCode(_graph);
  torch::jit::ConstantPropagation(_graph);
  torch::jit::EliminateCommonSubexpression(_graph);
  torch::jit::FuseGraph(_graph, true);

  // Create executor
  _executor = std::make_shared<torch::jit::GraphExecutor>(_graph, "F");
}

torch::Tensor
ParsedJITTensor::eval(const std::vector<const torch::Tensor *> & params)
{
  if (!_ast)
    throw std::runtime_error("No expression to evaluate");

  if (params.size() != _variables.size())
    throw std::runtime_error("Parameter count mismatch");

  // Compile if not already done
  if (!_executor)
    compile();

  // Build stack: first variables, then constants
  torch::jit::Stack stack;
  for (const auto & p : params)
    stack.push_back(*p);

  // Add constants to the stack
  for (const auto & kv : _constants)
    stack.push_back(kv.second);

  // Execute
  torch::NoGradGuard no_grad;
  _executor->run(stack);

  if (stack.size() != 1)
  {
    std::string msg = "Unexpected number of outputs: " + std::to_string(stack.size());
    for (size_t i = 0; i < stack.size(); ++i)
      msg += "\n  [" + std::to_string(i) + "]: " + stack[i].tagKind();
    throw std::runtime_error(msg);
  }

  // Handle case where JIT optimization collapsed expression to a scalar constant
  if (stack[0].isTensor())
    return stack[0].toTensor();
  else if (stack[0].isDouble())
    return torch::tensor(stack[0].toDouble(), MooseTensor::floatTensorOptions());
  else if (stack[0].isInt())
    return torch::tensor(stack[0].toInt(), MooseTensor::floatTensorOptions());
  else
    throw std::runtime_error("Unexpected output type from JIT executor");
}

void
ParsedJITTensor::print() const
{
  if (_graph)
    _graph->dump();
  else
    std::cout << "No compiled graph\n";
}

std::string
ParsedJITTensor::toString() const
{
  if (!_ast)
    return "<no expression>";
  return _ast->toString();
}
