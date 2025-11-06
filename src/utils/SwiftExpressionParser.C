/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SwiftExpressionParser.h"
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <stdexcept>
#include <sstream>
#include <cmath>

namespace SwiftExpressionParser
{

// Helper to check if an expression is a constant
static std::shared_ptr<Constant>
asConstant(const ExprPtr & expr)
{
  return std::dynamic_pointer_cast<Constant>(expr);
}

//=============================================================================
// BinaryOp implementation
//=============================================================================

const char *
BinaryOp::opString(Op op)
{
  switch (op)
  {
    case Op::Add:
      return "+";
    case Op::Sub:
      return "-";
    case Op::Mul:
      return "*";
    case Op::Div:
      return "/";
    case Op::Pow:
      return "^";
    case Op::Mod:
      return "%";
  }
  return "?";
}

ExprPtr
BinaryOp::simplify() const
{
  auto left = _left->simplify();
  auto right = _right->simplify();

  auto lc = asConstant(left);
  auto rc = asConstant(right);

  // Both constants - fold
  if (lc && rc)
  {
    double result = 0.0;
    switch (_op)
    {
      case Op::Add:
        result = lc->value() + rc->value();
        break;
      case Op::Sub:
        result = lc->value() - rc->value();
        break;
      case Op::Mul:
        result = lc->value() * rc->value();
        break;
      case Op::Div:
        result = lc->value() / rc->value();
        break;
      case Op::Pow:
        result = std::pow(lc->value(), rc->value());
        break;
      case Op::Mod:
        result = std::fmod(lc->value(), rc->value());
        break;
    }
    return std::make_shared<Constant>(result);
  }

  // Algebraic simplifications
  switch (_op)
  {
    case Op::Add:
      if (lc && lc->value() == 0.0)
        return right;
      if (rc && rc->value() == 0.0)
        return left;
      break;

    case Op::Sub:
      if (rc && rc->value() == 0.0)
        return left;
      if (lc && lc->value() == 0.0)
        return std::make_shared<UnaryOp>(UnaryOp::Op::Neg, right)->simplify();
      break;

    case Op::Mul:
      if (lc && lc->value() == 0.0)
        return std::make_shared<Constant>(0.0);
      if (rc && rc->value() == 0.0)
        return std::make_shared<Constant>(0.0);
      if (lc && lc->value() == 1.0)
        return right;
      if (rc && rc->value() == 1.0)
        return left;
      if (lc && lc->value() == -1.0)
        return std::make_shared<UnaryOp>(UnaryOp::Op::Neg, right)->simplify();
      if (rc && rc->value() == -1.0)
        return std::make_shared<UnaryOp>(UnaryOp::Op::Neg, left)->simplify();
      break;

    case Op::Div:
      if (lc && lc->value() == 0.0)
        return std::make_shared<Constant>(0.0);
      if (rc && rc->value() == 1.0)
        return left;
      break;

    case Op::Pow:
      if (rc && rc->value() == 0.0)
        return std::make_shared<Constant>(1.0);
      if (rc && rc->value() == 1.0)
        return left;
      if (lc && lc->value() == 1.0)
        return std::make_shared<Constant>(1.0);
      break;

    case Op::Mod:
      // No common algebraic simplifications for modulo
      break;
  }

  return std::make_shared<BinaryOp>(_op, left, right);
}

ExprPtr
BinaryOp::differentiate(const std::string & var) const
{
  auto dl = _left->differentiate(var);
  auto dr = _right->differentiate(var);

  switch (_op)
  {
    case Op::Add:
      return std::make_shared<BinaryOp>(Op::Add, dl, dr);

    case Op::Sub:
      return std::make_shared<BinaryOp>(Op::Sub, dl, dr);

    case Op::Mul:
      // (f*g)' = f'*g + f*g'
      return std::make_shared<BinaryOp>(Op::Add,
                                        std::make_shared<BinaryOp>(Op::Mul, dl, _right),
                                        std::make_shared<BinaryOp>(Op::Mul, _left, dr));

    case Op::Div:
      // (f/g)' = (f'*g - f*g') / g^2
      return std::make_shared<BinaryOp>(
          Op::Div,
          std::make_shared<BinaryOp>(Op::Sub,
                                     std::make_shared<BinaryOp>(Op::Mul, dl, _right),
                                     std::make_shared<BinaryOp>(Op::Mul, _left, dr)),
          std::make_shared<BinaryOp>(Op::Pow, _right, std::make_shared<Constant>(2.0)));

    case Op::Pow:
      // For f^g where g is constant: (f^g)' = g * f^(g-1) * f'
      if (auto rc = asConstant(_right))
      {
        return std::make_shared<BinaryOp>(
            Op::Mul,
            std::make_shared<BinaryOp>(
                Op::Mul,
                _right,
                std::make_shared<BinaryOp>(
                    Op::Pow, _left, std::make_shared<Constant>(rc->value() - 1.0))),
            dl);
      }
      // General case: (f^g)' = f^g * (g' * ln(f) + g * f'/f)
      return std::make_shared<BinaryOp>(
          Op::Mul,
          std::make_shared<BinaryOp>(Op::Pow, _left, _right),
          std::make_shared<BinaryOp>(
              Op::Add,
              std::make_shared<BinaryOp>(
                  Op::Mul, dr, std::make_shared<FunctionCall>("log", std::vector<ExprPtr>{_left})),
              std::make_shared<BinaryOp>(
                  Op::Mul, _right, std::make_shared<BinaryOp>(Op::Div, dl, _left))));

    case Op::Mod:
      // d/dvar[x % y] = dx (treating y as constant for now)
      // Note: Full derivative would be dx - y*d/dvar[floor(x/y)] which is discontinuous
      // For practical purposes, the derivative w.r.t. x is dx almost everywhere
      return dl;
  }
  throw std::runtime_error("Invalid binary operator");
}

ExprPtr
BinaryOp::substitute(const std::string & var, const ExprPtr & replacement) const
{
  return std::make_shared<BinaryOp>(
      _op, _left->substitute(var, replacement), _right->substitute(var, replacement));
}

torch::jit::Value *
BinaryOp::buildGraph(torch::jit::Graph & graph,
                     std::unordered_map<std::string, torch::jit::Value *> & vars) const
{
  auto left = _left->buildGraph(graph, vars);
  auto right = _right->buildGraph(graph, vars);

  switch (_op)
  {
    case Op::Add:
      return graph.insert(torch::jit::aten::add, {left, right});
    case Op::Sub:
      return graph.insert(torch::jit::aten::sub, {left, right});
    case Op::Mul:
      return graph.insert(torch::jit::aten::mul, {left, right});
    case Op::Div:
      return graph.insert(torch::jit::aten::div, {left, right});
    case Op::Pow:
      return graph.insert(torch::jit::aten::pow, {left, right});
    case Op::Mod:
      return graph.insert(torch::jit::aten::remainder, {left, right});
  }
  throw std::runtime_error("Invalid binary operator");
}

std::string
BinaryOp::toString() const
{
  return "(" + _left->toString() + " " + opString(_op) + " " + _right->toString() + ")";
}

//=============================================================================
// UnaryOp implementation
//=============================================================================

ExprPtr
UnaryOp::simplify() const
{
  auto operand = _operand->simplify();

  if (auto c = asConstant(operand))
  {
    switch (_op)
    {
      case Op::Neg:
        return std::make_shared<Constant>(-c->value());
      case Op::Not:
        return std::make_shared<Constant>(c->value() == 0.0 ? 1.0 : 0.0);
    }
  }

  return std::make_shared<UnaryOp>(_op, operand);
}

ExprPtr
UnaryOp::differentiate(const std::string & var) const
{
  switch (_op)
  {
    case Op::Neg:
      return std::make_shared<UnaryOp>(Op::Neg, _operand->differentiate(var));
    case Op::Not:
      return std::make_shared<Constant>(0.0); // Derivative of logical not is zero
  }
  throw std::runtime_error("Invalid unary operator");
}

ExprPtr
UnaryOp::substitute(const std::string & var, const ExprPtr & replacement) const
{
  return std::make_shared<UnaryOp>(_op, _operand->substitute(var, replacement));
}

torch::jit::Value *
UnaryOp::buildGraph(torch::jit::Graph & graph,
                    std::unordered_map<std::string, torch::jit::Value *> & vars) const
{
  auto operand = _operand->buildGraph(graph, vars);

  switch (_op)
  {
    case Op::Neg:
      return graph.insert(torch::jit::aten::neg, {operand});
    case Op::Not:
      return graph.insert(torch::jit::aten::logical_not, {operand});
  }
  throw std::runtime_error("Invalid unary operator");
}

std::string
UnaryOp::toString() const
{
  switch (_op)
  {
    case Op::Neg:
      return "(-" + _operand->toString() + ")";
    case Op::Not:
      return "(!" + _operand->toString() + ")";
  }
  return "?";
}

//=============================================================================
// Comparison implementation
//=============================================================================

ExprPtr
Comparison::simplify() const
{
  auto left = _left->simplify();
  auto right = _right->simplify();

  auto lc = asConstant(left);
  auto rc = asConstant(right);

  if (lc && rc)
  {
    bool result = false;
    switch (_op)
    {
      case Op::Lt:
        result = lc->value() < rc->value();
        break;
      case Op::Gt:
        result = lc->value() > rc->value();
        break;
      case Op::Le:
        result = lc->value() <= rc->value();
        break;
      case Op::Ge:
        result = lc->value() >= rc->value();
        break;
      case Op::Eq:
        result = lc->value() == rc->value();
        break;
      case Op::Ne:
        result = lc->value() != rc->value();
        break;
    }
    return std::make_shared<Constant>(result ? 1.0 : 0.0);
  }

  return std::make_shared<Comparison>(_op, left, right);
}

ExprPtr
Comparison::differentiate(const std::string &) const
{
  // Derivative of comparison is zero (not differentiable in classical sense)
  return std::make_shared<Constant>(0.0);
}

ExprPtr
Comparison::substitute(const std::string & var, const ExprPtr & replacement) const
{
  return std::make_shared<Comparison>(
      _op, _left->substitute(var, replacement), _right->substitute(var, replacement));
}

torch::jit::Value *
Comparison::buildGraph(torch::jit::Graph & graph,
                       std::unordered_map<std::string, torch::jit::Value *> & vars) const
{
  auto left = _left->buildGraph(graph, vars);
  auto right = _right->buildGraph(graph, vars);

  switch (_op)
  {
    case Op::Lt:
      return graph.insert(torch::jit::aten::lt, {left, right});
    case Op::Gt:
      return graph.insert(torch::jit::aten::gt, {left, right});
    case Op::Le:
      return graph.insert(torch::jit::aten::le, {left, right});
    case Op::Ge:
      return graph.insert(torch::jit::aten::ge, {left, right});
    case Op::Eq:
      return graph.insert(torch::jit::aten::eq, {left, right});
    case Op::Ne:
      return graph.insert(torch::jit::aten::ne, {left, right});
  }
  throw std::runtime_error("Invalid comparison operator");
}

std::string
Comparison::toString() const
{
  const char * op = "?";
  switch (_op)
  {
    case Op::Lt:
      op = "<";
      break;
    case Op::Gt:
      op = ">";
      break;
    case Op::Le:
      op = "<=";
      break;
    case Op::Ge:
      op = ">=";
      break;
    case Op::Eq:
      op = "==";
      break;
    case Op::Ne:
      op = "!=";
      break;
  }
  return "(" + _left->toString() + " " + op + " " + _right->toString() + ")";
}

//=============================================================================
// LogicalOp implementation
//=============================================================================

ExprPtr
LogicalOp::simplify() const
{
  auto left = _left->simplify();
  auto right = _right->simplify();

  auto lc = asConstant(left);
  auto rc = asConstant(right);

  if (lc && rc)
  {
    bool lv = lc->value() != 0.0;
    bool rv = rc->value() != 0.0;
    bool result = false;
    switch (_op)
    {
      case Op::And:
        result = lv && rv;
        break;
      case Op::Or:
        result = lv || rv;
        break;
    }
    return std::make_shared<Constant>(result ? 1.0 : 0.0);
  }

  // Algebraic simplifications
  switch (_op)
  {
    case Op::And:
      if (lc && lc->value() == 0.0)
        return std::make_shared<Constant>(0.0);
      if (rc && rc->value() == 0.0)
        return std::make_shared<Constant>(0.0);
      break;
    case Op::Or:
      if (lc && lc->value() != 0.0)
        return std::make_shared<Constant>(1.0);
      if (rc && rc->value() != 0.0)
        return std::make_shared<Constant>(1.0);
      break;
  }

  return std::make_shared<LogicalOp>(_op, left, right);
}

ExprPtr
LogicalOp::differentiate(const std::string &) const
{
  // Derivative of logical operations is zero
  return std::make_shared<Constant>(0.0);
}

ExprPtr
LogicalOp::substitute(const std::string & var, const ExprPtr & replacement) const
{
  return std::make_shared<LogicalOp>(
      _op, _left->substitute(var, replacement), _right->substitute(var, replacement));
}

torch::jit::Value *
LogicalOp::buildGraph(torch::jit::Graph & graph,
                      std::unordered_map<std::string, torch::jit::Value *> & vars) const
{
  auto left = _left->buildGraph(graph, vars);
  auto right = _right->buildGraph(graph, vars);

  switch (_op)
  {
    case Op::And:
      return graph.insert(torch::jit::aten::logical_and, {left, right});
    case Op::Or:
      return graph.insert(torch::jit::aten::logical_or, {left, right});
  }
  throw std::runtime_error("Invalid logical operator");
}

std::string
LogicalOp::toString() const
{
  const char * op = (_op == Op::And) ? "&" : "|";
  return "(" + _left->toString() + " " + op + " " + _right->toString() + ")";
}

//=============================================================================
// FunctionCall implementation
//=============================================================================

ExprPtr
FunctionCall::simplify() const
{
  // Simplify all arguments
  std::vector<ExprPtr> simplified_args;
  bool all_const = true;
  std::vector<double> const_values;

  for (const auto & arg : _args)
  {
    auto s = arg->simplify();
    simplified_args.push_back(s);
    if (auto c = asConstant(s))
      const_values.push_back(c->value());
    else
      all_const = false;
  }

  // If all arguments are constants, fold the function
  if (all_const)
  {
    double result = 0.0;
    if (_name == "sin" && const_values.size() == 1)
      result = std::sin(const_values[0]);
    else if (_name == "cos" && const_values.size() == 1)
      result = std::cos(const_values[0]);
    else if (_name == "tan" && const_values.size() == 1)
      result = std::tan(const_values[0]);
    else if (_name == "sinh" && const_values.size() == 1)
      result = std::sinh(const_values[0]);
    else if (_name == "cosh" && const_values.size() == 1)
      result = std::cosh(const_values[0]);
    else if (_name == "tanh" && const_values.size() == 1)
      result = std::tanh(const_values[0]);
    else if (_name == "asin" && const_values.size() == 1)
      result = std::asin(const_values[0]);
    else if (_name == "acos" && const_values.size() == 1)
      result = std::acos(const_values[0]);
    else if (_name == "atan" && const_values.size() == 1)
      result = std::atan(const_values[0]);
    else if (_name == "asinh" && const_values.size() == 1)
      result = std::asinh(const_values[0]);
    else if (_name == "acosh" && const_values.size() == 1)
      result = std::acosh(const_values[0]);
    else if (_name == "atanh" && const_values.size() == 1)
      result = std::atanh(const_values[0]);
    else if (_name == "exp" && const_values.size() == 1)
      result = std::exp(const_values[0]);
    else if (_name == "log" && const_values.size() == 1)
      result = std::log(const_values[0]);
    else if (_name == "log10" && const_values.size() == 1)
      result = std::log10(const_values[0]);
    else if (_name == "log2" && const_values.size() == 1)
      result = std::log2(const_values[0]);
    else if (_name == "sqrt" && const_values.size() == 1)
      result = std::sqrt(const_values[0]);
    else if (_name == "abs" && const_values.size() == 1)
      result = std::abs(const_values[0]);
    else if (_name == "ceil" && const_values.size() == 1)
      result = std::ceil(const_values[0]);
    else if (_name == "floor" && const_values.size() == 1)
      result = std::floor(const_values[0]);
    else if (_name == "round" && const_values.size() == 1)
      result = std::round(const_values[0]);
    else if (_name == "trunc" && const_values.size() == 1)
      result = std::trunc(const_values[0]);
    else if (_name == "min" && const_values.size() == 2)
      result = std::min(const_values[0], const_values[1]);
    else if (_name == "max" && const_values.size() == 2)
      result = std::max(const_values[0], const_values[1]);
    else if (_name == "atan2" && const_values.size() == 2)
      result = std::atan2(const_values[0], const_values[1]);
    else if (_name == "hypot" && const_values.size() == 2)
      result = std::hypot(const_values[0], const_values[1]);
    else if (_name == "pow" && const_values.size() == 2)
      result = std::pow(const_values[0], const_values[1]);
    else if (_name == "if" && const_values.size() == 3)
      result = const_values[0] != 0.0 ? const_values[1] : const_values[2];
    else
      return std::make_shared<FunctionCall>(_name, simplified_args);

    return std::make_shared<Constant>(result);
  }

  return std::make_shared<FunctionCall>(_name, simplified_args);
}

ExprPtr
FunctionCall::differentiate(const std::string & var) const
{
  if (_args.empty())
    return std::make_shared<Constant>(0.0);

  auto arg = _args[0];
  auto darg = arg->differentiate(var);

  // Chain rule for single-argument functions
  if (_name == "sin")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul, std::make_shared<FunctionCall>("cos", std::vector<ExprPtr>{arg}), darg);
  else if (_name == "cos")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul,
        std::make_shared<UnaryOp>(UnaryOp::Op::Neg,
                                  std::make_shared<FunctionCall>("sin", std::vector<ExprPtr>{arg})),
        darg);
  else if (_name == "tan")
  {
    auto cos_arg = std::make_shared<FunctionCall>("cos", std::vector<ExprPtr>{arg});
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Div, darg, std::make_shared<BinaryOp>(BinaryOp::Op::Mul, cos_arg, cos_arg));
  }
  else if (_name == "sinh")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul, std::make_shared<FunctionCall>("cosh", std::vector<ExprPtr>{arg}), darg);
  else if (_name == "cosh")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul, std::make_shared<FunctionCall>("sinh", std::vector<ExprPtr>{arg}), darg);
  else if (_name == "tanh")
  {
    auto cosh_arg = std::make_shared<FunctionCall>("cosh", std::vector<ExprPtr>{arg});
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Div, darg, std::make_shared<BinaryOp>(BinaryOp::Op::Mul, cosh_arg, cosh_arg));
  }
  else if (_name == "exp")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul, std::make_shared<FunctionCall>("exp", std::vector<ExprPtr>{arg}), darg);
  else if (_name == "exp2")
  {
    // d/dx[2^x] = 2^x * ln(2) * dx
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul,
        std::make_shared<BinaryOp>(
            BinaryOp::Op::Mul,
            std::make_shared<FunctionCall>("exp2", std::vector<ExprPtr>{arg}),
            std::make_shared<FunctionCall>("log",
                                           std::vector<ExprPtr>{std::make_shared<Constant>(2.0)})),
        darg);
  }
  else if (_name == "log")
    return std::make_shared<BinaryOp>(BinaryOp::Op::Div, darg, arg);
  else if (_name == "log10")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Div,
        darg,
        std::make_shared<BinaryOp>(
            BinaryOp::Op::Mul,
            arg,
            std::make_shared<FunctionCall>(
                "log", std::vector<ExprPtr>{std::make_shared<Constant>(10.0)})));
  else if (_name == "log2")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Div,
        darg,
        std::make_shared<BinaryOp>(
            BinaryOp::Op::Mul,
            arg,
            std::make_shared<FunctionCall>("log",
                                           std::vector<ExprPtr>{std::make_shared<Constant>(2.0)})));
  else if (_name == "sqrt")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Div,
        darg,
        std::make_shared<BinaryOp>(
            BinaryOp::Op::Mul,
            std::make_shared<Constant>(2.0),
            std::make_shared<FunctionCall>("sqrt", std::vector<ExprPtr>{arg})));
  else if (_name == "rsqrt")
  {
    // d/dx[1/sqrt(x)] = -1/(2*x^(3/2)) * dx = -rsqrt(x)/(2*x) * dx
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul,
        std::make_shared<UnaryOp>(
            UnaryOp::Op::Neg,
            std::make_shared<BinaryOp>(
                BinaryOp::Op::Div,
                std::make_shared<FunctionCall>("rsqrt", std::vector<ExprPtr>{arg}),
                std::make_shared<BinaryOp>(
                    BinaryOp::Op::Mul, std::make_shared<Constant>(2.0), arg))),
        darg);
  }
  else if (_name == "asin")
  {
    // d/dx[asin(x)] = 1/sqrt(1-x^2) * dx
    auto one_minus_x2 =
        std::make_shared<BinaryOp>(BinaryOp::Op::Sub,
                                   std::make_shared<Constant>(1.0),
                                   std::make_shared<BinaryOp>(BinaryOp::Op::Mul, arg, arg));
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Div,
        darg,
        std::make_shared<FunctionCall>("sqrt", std::vector<ExprPtr>{one_minus_x2}));
  }
  else if (_name == "acos")
  {
    // d/dx[acos(x)] = -1/sqrt(1-x^2) * dx
    auto one_minus_x2 =
        std::make_shared<BinaryOp>(BinaryOp::Op::Sub,
                                   std::make_shared<Constant>(1.0),
                                   std::make_shared<BinaryOp>(BinaryOp::Op::Mul, arg, arg));
    return std::make_shared<UnaryOp>(
        UnaryOp::Op::Neg,
        std::make_shared<BinaryOp>(
            BinaryOp::Op::Div,
            darg,
            std::make_shared<FunctionCall>("sqrt", std::vector<ExprPtr>{one_minus_x2})));
  }
  else if (_name == "atan")
  {
    // d/dx[atan(x)] = 1/(1+x^2) * dx
    auto one_plus_x2 =
        std::make_shared<BinaryOp>(BinaryOp::Op::Add,
                                   std::make_shared<Constant>(1.0),
                                   std::make_shared<BinaryOp>(BinaryOp::Op::Mul, arg, arg));
    return std::make_shared<BinaryOp>(BinaryOp::Op::Div, darg, one_plus_x2);
  }
  else if (_name == "asinh")
  {
    // d/dx[asinh(x)] = 1/sqrt(x^2+1) * dx
    auto x2_plus_one =
        std::make_shared<BinaryOp>(BinaryOp::Op::Add,
                                   std::make_shared<BinaryOp>(BinaryOp::Op::Mul, arg, arg),
                                   std::make_shared<Constant>(1.0));
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Div,
        darg,
        std::make_shared<FunctionCall>("sqrt", std::vector<ExprPtr>{x2_plus_one}));
  }
  else if (_name == "acosh")
  {
    // d/dx[acosh(x)] = 1/sqrt(x^2-1) * dx
    auto x2_minus_one =
        std::make_shared<BinaryOp>(BinaryOp::Op::Sub,
                                   std::make_shared<BinaryOp>(BinaryOp::Op::Mul, arg, arg),
                                   std::make_shared<Constant>(1.0));
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Div,
        darg,
        std::make_shared<FunctionCall>("sqrt", std::vector<ExprPtr>{x2_minus_one}));
  }
  else if (_name == "atanh")
  {
    // d/dx[atanh(x)] = 1/(1-x^2) * dx
    auto one_minus_x2 =
        std::make_shared<BinaryOp>(BinaryOp::Op::Sub,
                                   std::make_shared<Constant>(1.0),
                                   std::make_shared<BinaryOp>(BinaryOp::Op::Mul, arg, arg));
    return std::make_shared<BinaryOp>(BinaryOp::Op::Div, darg, one_minus_x2);
  }
  else if (_name == "abs")
  {
    // d/dx[abs(x)] = sign(x) * dx, but sign is discontinuous at 0
    // We use x/abs(x) which is undefined at 0 but gives correct derivatives elsewhere
    auto abs_val = std::make_shared<FunctionCall>(_name, _args);
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul, std::make_shared<BinaryOp>(BinaryOp::Op::Div, arg, abs_val), darg);
  }
  else if (_name == "hypot" && _args.size() == 2)
  {
    // d/dx[hypot(x,y)] = x/hypot(x,y) * dx + y/hypot(x,y) * dy
    auto arg2 = _args[1];
    auto darg2 = arg2->differentiate(var);
    auto hypot_val = std::make_shared<FunctionCall>(_name, _args);
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Add,
        std::make_shared<BinaryOp>(
            BinaryOp::Op::Mul, std::make_shared<BinaryOp>(BinaryOp::Op::Div, arg, hypot_val), darg),
        std::make_shared<BinaryOp>(BinaryOp::Op::Mul,
                                   std::make_shared<BinaryOp>(BinaryOp::Op::Div, arg2, hypot_val),
                                   darg2));
  }
  else if (_name == "atan2" && _args.size() == 2)
  {
    // d/dvar[atan2(y,x)] = (x*dy - y*dx)/(x^2+y^2)
    auto y = arg;
    auto x = _args[1];
    auto dy = darg;
    auto dx = _args[1]->differentiate(var);
    auto numerator =
        std::make_shared<BinaryOp>(BinaryOp::Op::Sub,
                                   std::make_shared<BinaryOp>(BinaryOp::Op::Mul, x, dy),
                                   std::make_shared<BinaryOp>(BinaryOp::Op::Mul, y, dx));
    auto denominator =
        std::make_shared<BinaryOp>(BinaryOp::Op::Add,
                                   std::make_shared<BinaryOp>(BinaryOp::Op::Mul, x, x),
                                   std::make_shared<BinaryOp>(BinaryOp::Op::Mul, y, y));
    return std::make_shared<BinaryOp>(BinaryOp::Op::Div, numerator, denominator);
  }
  else if (_name == "pow" && _args.size() == 2)
  {
    // d/dvar[x^y] = x^y * (y*dx/x + ln(x)*dy)
    // This is the full derivative accounting for both x and y being functions of var
    auto x = arg;
    auto y = _args[1];
    auto dx = darg;
    auto dy = _args[1]->differentiate(var);
    auto pow_val = std::make_shared<FunctionCall>(_name, _args);
    auto term1 = std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul, y, std::make_shared<BinaryOp>(BinaryOp::Op::Div, dx, x));
    auto term2 = std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul, std::make_shared<FunctionCall>("log", std::vector<ExprPtr>{x}), dy);
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul, pow_val, std::make_shared<BinaryOp>(BinaryOp::Op::Add, term1, term2));
  }
  else if (_name == "min" && _args.size() == 2)
  {
    // d/dvar[min(x,y)] = if(x<y, dx, dy)
    auto x = arg;
    auto y = _args[1];
    auto dx = darg;
    auto dy = _args[1]->differentiate(var);
    auto condition = std::make_shared<Comparison>(Comparison::Op::Lt, x, y);
    return std::make_shared<FunctionCall>("if", std::vector<ExprPtr>{condition, dx, dy});
  }
  else if (_name == "max" && _args.size() == 2)
  {
    // d/dvar[max(x,y)] = if(x>y, dx, dy)
    auto x = arg;
    auto y = _args[1];
    auto dx = darg;
    auto dy = _args[1]->differentiate(var);
    auto condition = std::make_shared<Comparison>(Comparison::Op::Gt, x, y);
    return std::make_shared<FunctionCall>("if", std::vector<ExprPtr>{condition, dx, dy});
  }
  else if (_name == "if" && _args.size() == 3)
  {
    // Derivative of if(c, t, f) w.r.t. var is if(c, dt/dvar, df/dvar)
    return std::make_shared<FunctionCall>(
        "if",
        std::vector<ExprPtr>{_args[0], _args[1]->differentiate(var), _args[2]->differentiate(var)});
  }

  // Functions with zero derivatives (discontinuous or constant on intervals)
  else if (_name == "round")
    return std::make_shared<Constant>(0.0); // d/dx[round(x)] = 0 almost everywhere
  else if (_name == "ceil")
    return std::make_shared<Constant>(0.0); // d/dx[ceil(x)] = 0 almost everywhere
  else if (_name == "floor")
    return std::make_shared<Constant>(0.0); // d/dx[floor(x)] = 0 almost everywhere
  else if (_name == "trunc")
    return std::make_shared<Constant>(0.0); // d/dx[trunc(x)] = 0 almost everywhere

  // If we reach here, derivative is not implemented for this function
  throw std::runtime_error("Derivative not implemented for function: " + _name);
}

ExprPtr
FunctionCall::substitute(const std::string & var, const ExprPtr & replacement) const
{
  std::vector<ExprPtr> substituted_args;
  for (const auto & arg : _args)
    substituted_args.push_back(arg->substitute(var, replacement));
  return std::make_shared<FunctionCall>(_name, substituted_args);
}

torch::jit::Value *
FunctionCall::buildGraph(torch::jit::Graph & graph,
                         std::unordered_map<std::string, torch::jit::Value *> & vars) const
{
  std::vector<torch::jit::Value *> arg_vals;
  for (const auto & arg : _args)
    arg_vals.push_back(arg->buildGraph(graph, vars));

  // Map function names to torch operations
  if (_name == "sin" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::sin, {arg_vals[0]});
  else if (_name == "cos" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::cos, {arg_vals[0]});
  else if (_name == "tan" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::tan, {arg_vals[0]});
  else if (_name == "sinh" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::sinh, {arg_vals[0]});
  else if (_name == "cosh" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::cosh, {arg_vals[0]});
  else if (_name == "tanh" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::tanh, {arg_vals[0]});
  else if (_name == "asin" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::asin, {arg_vals[0]});
  else if (_name == "acos" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::acos, {arg_vals[0]});
  else if (_name == "atan" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::atan, {arg_vals[0]});
  else if (_name == "asinh" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::asinh, {arg_vals[0]});
  else if (_name == "acosh" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::acosh, {arg_vals[0]});
  else if (_name == "atanh" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::atanh, {arg_vals[0]});
  else if (_name == "exp" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::exp, {arg_vals[0]});
  else if (_name == "exp2" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::exp2, {arg_vals[0]});
  else if (_name == "log" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::log, {arg_vals[0]});
  else if (_name == "log10" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::log10, {arg_vals[0]});
  else if (_name == "log2" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::log2, {arg_vals[0]});
  else if (_name == "sqrt" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::sqrt, {arg_vals[0]});
  else if (_name == "rsqrt" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::rsqrt, {arg_vals[0]});
  else if (_name == "abs" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::abs, {arg_vals[0]});
  else if (_name == "ceil" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::ceil, {arg_vals[0]});
  else if (_name == "floor" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::floor, {arg_vals[0]});
  else if (_name == "round" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::round, {arg_vals[0]});
  else if (_name == "trunc" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::trunc, {arg_vals[0]});
  else if (_name == "min" && arg_vals.size() == 2)
    return graph.insert(torch::jit::aten::minimum, {arg_vals[0], arg_vals[1]});
  else if (_name == "max" && arg_vals.size() == 2)
    return graph.insert(torch::jit::aten::maximum, {arg_vals[0], arg_vals[1]});
  else if (_name == "atan2" && arg_vals.size() == 2)
    return graph.insert(torch::jit::aten::atan2, {arg_vals[0], arg_vals[1]});
  else if (_name == "hypot" && arg_vals.size() == 2)
    return graph.insert(torch::jit::aten::hypot, {arg_vals[0], arg_vals[1]});
  else if (_name == "pow" && arg_vals.size() == 2)
    return graph.insert(torch::jit::aten::pow, {arg_vals[0], arg_vals[1]});
  else if (_name == "if" && arg_vals.size() == 3)
    return graph.insert(torch::jit::aten::where, {arg_vals[0], arg_vals[1], arg_vals[2]});
  else if (_name == "FFT" && arg_vals.size() == 1)
  {
    // Handle FFT specially - need to determine dimensions
    auto input = arg_vals[0];
    // For now, we'll use rfft (1D), but this should be determined based on input dimensions
    return graph.insert(torch::jit::aten::fft_rfft, {input});
  }
  else if (_name == "iFFT" && arg_vals.size() == 1)
  {
    auto input = arg_vals[0];
    return graph.insert(torch::jit::aten::fft_irfft, {input});
  }

  throw std::runtime_error("Unknown or unsupported function: " + _name);
}

std::string
FunctionCall::toString() const
{
  std::string result = _name + "(";
  for (size_t i = 0; i < _args.size(); ++i)
  {
    if (i > 0)
      result += ", ";
    result += _args[i]->toString();
  }
  result += ")";
  return result;
}

//=============================================================================
// LetExpression implementation
//=============================================================================

ExprPtr
LetExpression::simplify() const
{
  // Simplify all bindings and the body
  std::vector<std::pair<std::string, ExprPtr>> simplified_bindings;
  for (const auto & [name, expr] : _bindings)
    simplified_bindings.push_back({name, expr->simplify()});

  auto simplified_body = _body->simplify();

  // If there are no bindings, just return the body
  if (simplified_bindings.empty())
    return simplified_body;

  return std::make_shared<LetExpression>(std::move(simplified_bindings), simplified_body);
}

ExprPtr
LetExpression::substitute(const std::string & var, const ExprPtr & replacement) const
{
  // Check if var is shadowed by any binding
  for (const auto & [name, _] : _bindings)
  {
    if (name == var)
    {
      // Variable is shadowed, don't substitute in the body
      // But we still need to substitute in the binding expressions
      std::vector<std::pair<std::string, ExprPtr>> new_bindings;
      for (const auto & [bind_name, bind_expr] : _bindings)
        new_bindings.push_back({bind_name, bind_expr->substitute(var, replacement)});
      return std::make_shared<LetExpression>(std::move(new_bindings), _body);
    }
  }

  // Variable is not shadowed, substitute everywhere
  std::vector<std::pair<std::string, ExprPtr>> new_bindings;
  for (const auto & [name, expr] : _bindings)
    new_bindings.push_back({name, expr->substitute(var, replacement)});

  return std::make_shared<LetExpression>(std::move(new_bindings),
                                         _body->substitute(var, replacement));
}

ExprPtr
LetExpression::differentiate(const std::string & var) const
{
  // To differentiate (let x1=e1, x2=e2, ... in body) w.r.t. var:
  // Use the chain rule: d/dvar[body(x1, x2, ...)] = ∂body/∂var + Σ(∂body/∂xi * dxi/dvar)
  //
  // We create a new LetExpression with:
  // 1. Original bindings: x1=e1, x2=e2, ...
  // 2. Derivative bindings: dx1=d/dvar[e1], dx2=d/dvar[e2], ...
  // 3. Body derivative using chain rule
  //
  // This preserves common subexpressions and avoids exponential expansion.
  //
  // Important: Bindings can reference earlier bindings (e.g., x2:=x^2; sinx2:=sin(x2))
  // So when computing derivatives of bindings, we must apply the chain rule for
  // dependencies on earlier local variables.

  // Build new bindings with both original and derivative bindings
  std::vector<std::pair<std::string, ExprPtr>> new_bindings;
  std::vector<std::string> binding_names; // Track binding names for chain rule

  for (const auto & [name, expr] : _bindings)
  {
    // Keep original binding
    new_bindings.push_back({name, expr});

    // Compute derivative: d/dvar[expr]
    // Start with partial derivative w.r.t. var
    ExprPtr derivative_expr = expr->differentiate(var);

    // Apply chain rule for dependencies on earlier local variables
    // For each earlier binding xi, add: ∂expr/∂xi * dxi
    for (const auto & earlier_name : binding_names)
    {
      ExprPtr partial_expr_xi = expr->differentiate(earlier_name);

      // Check if this partial is non-zero (optimization)
      if (auto c = asConstant(partial_expr_xi))
      {
        if (c->value() == 0.0)
          continue; // Skip if partial derivative is zero
      }

      // Add chain rule term: ∂expr/∂xi * dxi
      std::string earlier_derivative_name = "d" + earlier_name;
      ExprPtr chain_term = std::make_shared<BinaryOp>(
          BinaryOp::Op::Mul, partial_expr_xi, std::make_shared<Variable>(earlier_derivative_name));

      // Accumulate: derivative_expr = derivative_expr + chain_term
      derivative_expr = std::make_shared<BinaryOp>(BinaryOp::Op::Add, derivative_expr, chain_term);
    }

    // Add derivative binding
    std::string derivative_name = "d" + name;
    new_bindings.push_back({derivative_name, derivative_expr});

    // Track this binding name for future chain rule applications
    binding_names.push_back(name);
  }

  // Compute the partial derivative of body w.r.t. var (treating local variables as constants)
  ExprPtr dbody = _body->differentiate(var);

  // Apply chain rule: for each local variable xi, add ∂body/∂xi * dxi
  for (const auto & [name, expr] : _bindings)
  {
    // Compute ∂body/∂xi (partial derivative w.r.t. local variable xi)
    ExprPtr partial_body_xi = _body->differentiate(name);

    // Check if this partial is non-zero (optimization)
    if (auto c = asConstant(partial_body_xi))
    {
      if (c->value() == 0.0)
        continue; // Skip if partial derivative is zero
    }

    // Add chain rule term: ∂body/∂xi * dxi
    std::string derivative_name = "d" + name;
    ExprPtr chain_term = std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul, partial_body_xi, std::make_shared<Variable>(derivative_name));

    // Accumulate: dbody = dbody + chain_term
    dbody = std::make_shared<BinaryOp>(BinaryOp::Op::Add, dbody, chain_term);
  }

  // Return new LetExpression with all bindings and the chain-rule derivative
  return std::make_shared<LetExpression>(std::move(new_bindings), dbody);
}

torch::jit::Value *
LetExpression::buildGraph(torch::jit::Graph & graph,
                          std::unordered_map<std::string, torch::jit::Value *> & vars) const
{
  // Create a scoped copy of vars
  auto scoped_vars = vars;

  // Evaluate bindings in order and add to scoped environment
  for (const auto & [name, expr] : _bindings)
  {
    auto value = expr->buildGraph(graph, scoped_vars);
    scoped_vars[name] = value;
  }

  // Evaluate body in the augmented environment
  return _body->buildGraph(graph, scoped_vars);
}

std::string
LetExpression::toString() const
{
  std::string result;
  for (const auto & [name, expr] : _bindings)
    result += name + ":=" + expr->toString() + "; ";
  result += _body->toString();
  return result;
}

//=============================================================================
// Parser implementation
//=============================================================================

Parser::Parser()
{
  if (!_parser.load_grammar(grammar))
    throw std::runtime_error("Failed to load expression grammar");

  // Set up semantic actions to build AST
  _parser["NUMBER"] = [](const peg::SemanticValues & sv) -> ExprPtr
  { return std::make_shared<Constant>(std::stod(sv.token_to_string())); };

  _parser["VARIABLE"] = [this](const peg::SemanticValues & sv) -> ExprPtr
  {
    std::string name = sv.token_to_string();
    // Trim whitespace
    name.erase(0, name.find_first_not_of(" \t\r\n"));
    name.erase(name.find_last_not_of(" \t\r\n") + 1);

    // Check if this is a constant
    if (_constants.count(name))
      return std::make_shared<ConstantTensor>(name);

    return std::make_shared<Variable>(name);
  };

  _parser["IDENTIFIER"] = [](const peg::SemanticValues & sv)
  {
    std::string name = sv.token_to_string();
    // Trim whitespace
    name.erase(0, name.find_first_not_of(" \t\r\n"));
    name.erase(name.find_last_not_of(" \t\r\n") + 1);
    return name;
  };

  _parser["PRIMARY"] = [](const peg::SemanticValues & sv) -> ExprPtr
  { return std::any_cast<ExprPtr>(sv[0]); };

  _parser["POWER"] = [](const peg::SemanticValues & sv) -> ExprPtr
  {
    ExprPtr result = std::any_cast<ExprPtr>(sv[0]);
    if (sv.size() > 1)
      result = std::make_shared<BinaryOp>(BinaryOp::Op::Pow, result, std::any_cast<ExprPtr>(sv[1]));
    return result;
  };

  _parser["UNARY_OP"] = [](const peg::SemanticValues & sv) { return sv.token_to_string(); };

  _parser["UNARY"] = [](const peg::SemanticValues & sv) -> ExprPtr
  {
    if (sv.size() == 1)
      return std::any_cast<ExprPtr>(sv[0]);

    std::string op = std::any_cast<std::string>(sv[0]);
    ExprPtr operand = std::any_cast<ExprPtr>(sv[1]);

    if (op == "-")
      return std::make_shared<UnaryOp>(UnaryOp::Op::Neg, operand);
    else if (op == "!")
      return std::make_shared<UnaryOp>(UnaryOp::Op::Not, operand);

    throw std::runtime_error("Unknown unary operator: " + op);
  };

  _parser["MUL_OP"] = [](const peg::SemanticValues & sv) { return sv.token_to_string(); };

  _parser["MULTITIVE"] = [](const peg::SemanticValues & sv) -> ExprPtr
  {
    ExprPtr result = std::any_cast<ExprPtr>(sv[0]);
    for (size_t i = 1; i < sv.size(); i += 2)
    {
      std::string op = std::any_cast<std::string>(sv[i]);
      ExprPtr right = std::any_cast<ExprPtr>(sv[i + 1]);

      if (op == "*")
        result = std::make_shared<BinaryOp>(BinaryOp::Op::Mul, result, right);
      else if (op == "/")
        result = std::make_shared<BinaryOp>(BinaryOp::Op::Div, result, right);
      else if (op == "%")
        result = std::make_shared<BinaryOp>(BinaryOp::Op::Mod, result, right);
    }
    return result;
  };

  _parser["ADD_OP"] = [](const peg::SemanticValues & sv) { return sv.token_to_string(); };

  _parser["ADDITIVE"] = [](const peg::SemanticValues & sv) -> ExprPtr
  {
    ExprPtr result = std::any_cast<ExprPtr>(sv[0]);
    for (size_t i = 1; i < sv.size(); i += 2)
    {
      std::string op = std::any_cast<std::string>(sv[i]);
      ExprPtr right = std::any_cast<ExprPtr>(sv[i + 1]);

      if (op == "+")
        result = std::make_shared<BinaryOp>(BinaryOp::Op::Add, result, right);
      else if (op == "-")
        result = std::make_shared<BinaryOp>(BinaryOp::Op::Sub, result, right);
    }
    return result;
  };

  _parser["COMP_OP"] = [](const peg::SemanticValues & sv) { return sv.token_to_string(); };

  _parser["COMPARISON"] = [](const peg::SemanticValues & sv) -> ExprPtr
  {
    if (sv.size() == 1)
      return std::any_cast<ExprPtr>(sv[0]);

    ExprPtr left = std::any_cast<ExprPtr>(sv[0]);
    std::string op = std::any_cast<std::string>(sv[1]);
    ExprPtr right = std::any_cast<ExprPtr>(sv[2]);

    Comparison::Op cmp_op;
    if (op == "<")
      cmp_op = Comparison::Op::Lt;
    else if (op == ">")
      cmp_op = Comparison::Op::Gt;
    else if (op == "<=")
      cmp_op = Comparison::Op::Le;
    else if (op == ">=")
      cmp_op = Comparison::Op::Ge;
    else if (op == "==")
      cmp_op = Comparison::Op::Eq;
    else if (op == "!=")
      cmp_op = Comparison::Op::Ne;
    else
      throw std::runtime_error("Unknown comparison operator: " + op);

    return std::make_shared<Comparison>(cmp_op, left, right);
  };

  _parser["LOGICAL_OP"] = [](const peg::SemanticValues & sv) { return sv.token_to_string(); };

  _parser["LOGICAL"] = [](const peg::SemanticValues & sv) -> ExprPtr
  {
    ExprPtr result = std::any_cast<ExprPtr>(sv[0]);
    for (size_t i = 1; i < sv.size(); i += 2)
    {
      std::string op = std::any_cast<std::string>(sv[i]);
      ExprPtr right = std::any_cast<ExprPtr>(sv[i + 1]);

      if (op == "&")
        result = std::make_shared<LogicalOp>(LogicalOp::Op::And, result, right);
      else if (op == "|")
        result = std::make_shared<LogicalOp>(LogicalOp::Op::Or, result, right);
    }
    return result;
  };

  _parser["ARGS"] = [](const peg::SemanticValues & sv)
  {
    std::vector<ExprPtr> args;
    for (const auto & v : sv)
      args.push_back(std::any_cast<ExprPtr>(v));
    return args;
  };

  _parser["FUNCTION"] = [](const peg::SemanticValues & sv) -> ExprPtr
  {
    std::string name = std::any_cast<std::string>(sv[0]);
    std::vector<ExprPtr> args;
    if (sv.size() > 1)
      args = std::any_cast<std::vector<ExprPtr>>(sv[1]);
    return std::make_shared<FunctionCall>(name, args);
  };

  _parser["ASSIGNMENT"] = [](const peg::SemanticValues & sv)
  {
    std::string name = std::any_cast<std::string>(sv[0]);
    ExprPtr expr = std::any_cast<ExprPtr>(sv[1]);
    return std::make_pair(name, expr);
  };

  _parser["STATEMENTS"] = [](const peg::SemanticValues & sv) -> ExprPtr
  {
    // sv contains: [assignment1, assignment2, ..., body_expr]
    // All but the last element are assignments

    if (sv.size() == 1)
    {
      // No assignments, just the body expression
      return std::any_cast<ExprPtr>(sv[0]);
    }

    // Extract bindings (all but last element)
    std::vector<std::pair<std::string, ExprPtr>> bindings;
    for (size_t i = 0; i < sv.size() - 1; ++i)
      bindings.push_back(std::any_cast<std::pair<std::string, ExprPtr>>(sv[i]));

    // Last element is the body
    ExprPtr body = std::any_cast<ExprPtr>(sv[sv.size() - 1]);

    return std::make_shared<LetExpression>(std::move(bindings), body);
  };

  _parser["EXPRESSION"] = [](const peg::SemanticValues & sv) -> ExprPtr
  { return std::any_cast<ExprPtr>(sv[0]); };
}

ExprPtr
Parser::parse(const std::string & expr, const std::unordered_set<std::string> & constants)
{
  _constants = constants;
  ExprPtr result;
  _error.clear();

  _parser.set_logger(
      [this](size_t line, size_t col, const std::string & msg, const std::string &)
      { _error = "Line " + std::to_string(line) + ":" + std::to_string(col) + ": " + msg; });

  if (!_parser.parse(expr, result))
  {
    if (_error.empty())
      _error = "Failed to parse expression";
    return nullptr;
  }

  return result;
}

} // namespace SwiftExpressionParser
