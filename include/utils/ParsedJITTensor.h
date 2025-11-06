/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "SwiftExpressionParser.h"
#include <torch/csrc/jit/ir/ir.h>
#include <torch/torch.h>

namespace torch
{
namespace jit
{
struct Graph;
struct GraphExecutor;
struct Value;
}
}

/**
 * ParsedJITTensor - JIT-optimized mathematical expression evaluator
 *
 * Parses mathematical expressions and compiles them to optimized PyTorch compute graphs
 * for efficient repeated evaluations with automatic differentiation support.
 */
class ParsedJITTensor
{
public:
  ParsedJITTensor();

  /// Parse an expression with given variable names and optional constants
  bool parse(const std::string & expression,
             const std::vector<std::string> & variables,
             const std::unordered_map<std::string, torch::Tensor> & constants = {});

  /// Take derivative with respect to a variable
  void differentiate(const std::string & var);

  /// Optimize and compile the expression
  void compile();

  /// Evaluate the expression with torch tensors
  torch::Tensor eval(const std::vector<const torch::Tensor *> & params);

  /// Get the error message from the last operation
  const std::string & errorMessage() const { return _error; }

  /// Print the IR graph for debugging
  void print() const;

  /// Get a string representation of the parsed expression
  std::string toString() const;

protected:
  SwiftExpressionParser::Parser _parser;
  SwiftExpressionParser::ExprPtr _ast;
  std::vector<std::string> _variables;
  std::unordered_map<std::string, torch::Tensor> _constants;
  std::string _error;

  /// Compiled graph
  std::shared_ptr<torch::jit::Graph> _graph;
  std::shared_ptr<torch::jit::GraphExecutor> _executor;
};
