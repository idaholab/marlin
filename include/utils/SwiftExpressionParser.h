/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "peglib.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <torch/torch.h>
#include <torch/csrc/jit/ir/ir.h>

/**
 * SwiftExpressionParser - A PEG-based mathematical expression parser for Swift
 *
 * This namespace contains a complete expression parser built on cpp-peglib that:
 * - Parses mathematical expressions into an Abstract Syntax Tree (AST)
 * - Supports binary operators: + - * / ^ (power)
 * - Supports unary operators: - (negation) ! (logical not)
 * - Supports comparison operators: < > <= >= == !=
 * - Supports logical operators: & (and) | (or)
 * - Supports parentheses for precedence control
 * - Supports mathematical functions (sin, cos, exp, log, sqrt, abs, etc.)
 * - Supports the if(condition, true_value, false_value) ternary function
 * - Performs symbolic differentiation
 * - Performs algebraic simplification and constant folding
 * - Generates optimized PyTorch JIT compute graphs
 *
 * The parser uses a clean separation of concerns:
 * 1. PEG grammar defines the syntax
 * 2. AST represents the parsed expression
 * 3. Simplification optimizes the AST
 * 4. Differentiation computes symbolic derivatives
 * 5. Graph builder generates PyTorch JIT IR
 */
namespace SwiftExpressionParser
{

// Forward declarations
class Expr;
using ExprPtr = std::shared_ptr<Expr>;

/**
 * Base class for all expression AST nodes
 */
class Expr
{
public:
  virtual ~Expr() = default;

  /// Simplify the expression (constant folding, algebraic simplification)
  virtual ExprPtr simplify() const = 0;

  /// Differentiate with respect to a variable
  virtual ExprPtr differentiate(const std::string & var) const = 0;

  /// Substitute a variable with an expression
  virtual ExprPtr substitute(const std::string & var, const ExprPtr & replacement) const = 0;

  /// Build a torch JIT graph node
  virtual torch::jit::Value *
  buildGraph(torch::jit::Graph & graph,
             std::unordered_map<std::string, torch::jit::Value *> & vars) const = 0;

  /// Debug string representation
  virtual std::string toString() const = 0;
};

/**
 * Constant value
 */
class Constant : public Expr
{
public:
  explicit Constant(double value) : _value(value) {}

  double value() const { return _value; }

  ExprPtr simplify() const override { return std::make_shared<Constant>(_value); }

  ExprPtr differentiate(const std::string &) const override
  {
    return std::make_shared<Constant>(0.0);
  }

  ExprPtr substitute(const std::string &, const ExprPtr &) const override
  {
    return std::make_shared<Constant>(_value);
  }

  torch::jit::Value *
  buildGraph(torch::jit::Graph & graph,
             std::unordered_map<std::string, torch::jit::Value *> &) const override
  {
    return graph.insertConstant(_value);
  }

  std::string toString() const override { return std::to_string(_value); }

private:
  double _value;
};

/**
 * Variable reference
 */
class Variable : public Expr
{
public:
  explicit Variable(const std::string & name) : _name(name) {}

  const std::string & name() const { return _name; }

  ExprPtr simplify() const override { return std::make_shared<Variable>(_name); }

  ExprPtr differentiate(const std::string & var) const override
  {
    return std::make_shared<Constant>(var == _name ? 1.0 : 0.0);
  }

  ExprPtr substitute(const std::string & var, const ExprPtr & replacement) const override
  {
    return (var == _name) ? replacement : std::make_shared<Variable>(_name);
  }

  torch::jit::Value *
  buildGraph(torch::jit::Graph &,
             std::unordered_map<std::string, torch::jit::Value *> & vars) const override
  {
    auto it = vars.find(_name);
    if (it == vars.end())
      throw std::runtime_error("Variable '" + _name + "' not found in variable list");
    return it->second;
  }

  std::string toString() const override { return _name; }

private:
  std::string _name;
};

/**
 * Constant tensor (for predefined constants like pi, e, i)
 */
class ConstantTensor : public Expr
{
public:
  explicit ConstantTensor(const std::string & name) : _name(name) {}

  const std::string & name() const { return _name; }

  ExprPtr simplify() const override { return std::make_shared<ConstantTensor>(_name); }

  ExprPtr differentiate(const std::string &) const override
  {
    return std::make_shared<Constant>(0.0);
  }

  ExprPtr substitute(const std::string &, const ExprPtr &) const override
  {
    return std::make_shared<ConstantTensor>(_name);
  }

  torch::jit::Value *
  buildGraph(torch::jit::Graph &,
             std::unordered_map<std::string, torch::jit::Value *> & vars) const override
  {
    auto it = vars.find(_name);
    if (it == vars.end())
      throw std::runtime_error("Constant '" + _name + "' not found in constants map");
    return it->second;
  }

  std::string toString() const override { return _name; }

private:
  std::string _name;
};

/**
 * Binary operations: +, -, *, /, ^
 */
class BinaryOp : public Expr
{
public:
  enum class Op
  {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod
  };

  BinaryOp(Op op, ExprPtr left, ExprPtr right) : _op(op), _left(left), _right(right) {}

  ExprPtr simplify() const override;
  ExprPtr differentiate(const std::string & var) const override;
  ExprPtr substitute(const std::string & var, const ExprPtr & replacement) const override;
  torch::jit::Value *
  buildGraph(torch::jit::Graph & graph,
             std::unordered_map<std::string, torch::jit::Value *> & vars) const override;
  std::string toString() const override;

private:
  Op _op;
  ExprPtr _left;
  ExprPtr _right;

  static const char * opString(Op op);
};

/**
 * Unary operations: -, !
 */
class UnaryOp : public Expr
{
public:
  enum class Op
  {
    Neg,
    Not
  };

  UnaryOp(Op op, ExprPtr operand) : _op(op), _operand(operand) {}

  ExprPtr simplify() const override;
  ExprPtr differentiate(const std::string & var) const override;
  ExprPtr substitute(const std::string & var, const ExprPtr & replacement) const override;
  torch::jit::Value *
  buildGraph(torch::jit::Graph & graph,
             std::unordered_map<std::string, torch::jit::Value *> & vars) const override;
  std::string toString() const override;

private:
  Op _op;
  ExprPtr _operand;
};

/**
 * Comparison operations: <, >, <=, >=, ==, !=
 */
class Comparison : public Expr
{
public:
  enum class Op
  {
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne
  };

  Comparison(Op op, ExprPtr left, ExprPtr right) : _op(op), _left(left), _right(right) {}

  ExprPtr simplify() const override;
  ExprPtr differentiate(const std::string & var) const override;
  ExprPtr substitute(const std::string & var, const ExprPtr & replacement) const override;
  torch::jit::Value *
  buildGraph(torch::jit::Graph & graph,
             std::unordered_map<std::string, torch::jit::Value *> & vars) const override;
  std::string toString() const override;

private:
  Op _op;
  ExprPtr _left;
  ExprPtr _right;
};

/**
 * Logical operations: &, |
 */
class LogicalOp : public Expr
{
public:
  enum class Op
  {
    And,
    Or
  };

  LogicalOp(Op op, ExprPtr left, ExprPtr right) : _op(op), _left(left), _right(right) {}

  ExprPtr simplify() const override;
  ExprPtr differentiate(const std::string & var) const override;
  ExprPtr substitute(const std::string & var, const ExprPtr & replacement) const override;
  torch::jit::Value *
  buildGraph(torch::jit::Graph & graph,
             std::unordered_map<std::string, torch::jit::Value *> & vars) const override;
  std::string toString() const override;

private:
  Op _op;
  ExprPtr _left;
  ExprPtr _right;
};

/**
 * Function call
 */
class FunctionCall : public Expr
{
public:
  FunctionCall(const std::string & name, std::vector<ExprPtr> args)
    : _name(name), _args(std::move(args))
  {
  }

  const std::string & name() const { return _name; }
  const std::vector<ExprPtr> & args() const { return _args; }

  ExprPtr simplify() const override;
  ExprPtr differentiate(const std::string & var) const override;
  ExprPtr substitute(const std::string & var, const ExprPtr & replacement) const override;
  torch::jit::Value *
  buildGraph(torch::jit::Graph & graph,
             std::unordered_map<std::string, torch::jit::Value *> & vars) const override;
  std::string toString() const override;

private:
  std::string _name;
  std::vector<ExprPtr> _args;
};

/**
 * Let expression with local variable bindings
 * Syntax: var1 := expr1; var2 := expr2; body_expression
 */
class LetExpression : public Expr
{
public:
  LetExpression(std::vector<std::pair<std::string, ExprPtr>> bindings, ExprPtr body)
    : _bindings(std::move(bindings)), _body(std::move(body))
  {
  }

  const std::vector<std::pair<std::string, ExprPtr>> & bindings() const { return _bindings; }
  const ExprPtr & body() const { return _body; }

  ExprPtr simplify() const override;
  ExprPtr differentiate(const std::string & var) const override;
  ExprPtr substitute(const std::string & var, const ExprPtr & replacement) const override;
  torch::jit::Value *
  buildGraph(torch::jit::Graph & graph,
             std::unordered_map<std::string, torch::jit::Value *> & vars) const override;
  std::string toString() const override;

private:
  std::vector<std::pair<std::string, ExprPtr>> _bindings;
  ExprPtr _body;
};

/**
 * Main parser class using peglib
 */
class Parser
{
public:
  Parser();

  /// Parse an expression string with optional set of constant names
  ExprPtr parse(const std::string & expr, const std::unordered_set<std::string> & constants = {});

  /// Get last error message
  const std::string & errorMessage() const { return _error; }

private:
  peg::parser _parser;
  std::string _error;
  std::unordered_set<std::string> _constants;

  static constexpr const char * grammar = R"(
    # Top-level expression (may include local variable assignments)
    EXPRESSION  <- STATEMENTS

    # Statements: assignments followed by final expression
    STATEMENTS  <- (ASSIGNMENT ';')* LOGICAL
    ASSIGNMENT  <- IDENTIFIER ':=' LOGICAL

    # Logical operators (lowest precedence)
    LOGICAL     <- COMPARISON (LOGICAL_OP COMPARISON)*
    LOGICAL_OP  <- '|' / '&'

    # Comparison operators
    COMPARISON  <- ADDITIVE (COMP_OP ADDITIVE)?
    COMP_OP     <- '<=' / '>=' / '==' / '!=' / '<' / '>'

    # Addition and subtraction
    ADDITIVE    <- MULTITIVE (ADD_OP MULTITIVE)*
    ADD_OP      <- '+' / '-'

    # Multiplication, division, and modulo
    MULTITIVE   <- UNARY (MUL_OP UNARY)*
    MUL_OP      <- '*' / '/' / '%'

    # Unary operators
    UNARY       <- (UNARY_OP UNARY) / POWER
    UNARY_OP    <- '-' / '!'

    # Power (right-associative)
    POWER       <- PRIMARY ('^' POWER)?

    # Primary expressions
    PRIMARY     <- FUNCTION / NUMBER / VARIABLE / '(' LOGICAL ')'

    # Function calls
    FUNCTION    <- IDENTIFIER '(' ARGS? ')'
    ARGS        <- LOGICAL (',' LOGICAL)*

    # Terminals
    NUMBER      <- < [0-9]+ ('.' [0-9]+)? ([eE] [+-]? [0-9]+)? >
    IDENTIFIER  <- < [a-zA-Z_][a-zA-Z0-9_]* >
    VARIABLE    <- IDENTIFIER !('(' / ':')

    %whitespace <- [ \t\r\n]*
  )";
};

} // namespace SwiftExpressionParser
