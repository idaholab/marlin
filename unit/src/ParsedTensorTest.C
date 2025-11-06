/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ParsedJITTensor.h"
#include "SwiftExpressionParser.h"
#include "SwiftUtils.h"
#include "MooseError.h"
#include "gtest/gtest.h"

#include <ATen/ops/linspace.h>
#include <string>
#include <vector>

TEST(ParsedTensorTest, Parse)
{
  // Use linspace values that avoid zero to prevent singularities in derivatives
  const auto x =
      torch::unsqueeze(torch::linspace(0.1, 2.01, 11, MooseTensor::floatTensorOptions()), 1);
  const auto y =
      torch::unsqueeze(torch::linspace(0.11, 3.02, 15, MooseTensor::floatTensorOptions()), 0);
  const auto n = torch::max(x * y) * 1.01;

  // Epsilon for precision-dependent testing
  const Real epsilon = MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-12 : 1e-6;

  // Epsilon tensor for finite difference calculations (use sqrt(epsilon) for better numerical
  // stability)
  const Real eps_fd = MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-6 : 1e-3;
  const Real rel_tolerance =
      MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-5 : 1e-2;
  const Real abs_tolerance = std::sqrt(epsilon);

  // Perturbed tensors for finite difference derivatives
  const auto x2 = x + eps_fd;
  const auto y2 = y + eps_fd;
  const auto n2 = n + eps_fd;

  auto check = [&](const std::string & expression, auto gold, bool derivative_check = true)
  {
    ParsedJITTensor fp_jit;
    std::vector<std::string> variables{"x", "y", "n"};

    if (!fp_jit.parse(expression, variables))
      mooseError("Invalid JIT function: ", expression, "   ", fp_jit.errorMessage());

    std::vector<const torch::Tensor *> params{&x, &y, &n};

    std::cout << "--  " << expression << std::endl;

    // Test without compilation
    const auto result_no_opt = fp_jit.eval(params);

    // Test with compilation (optimization)
    ParsedJITTensor fp_jit_opt;
    fp_jit_opt.parse(expression, variables);
    fp_jit_opt.compile(); // compile does optimization for JIT

    const auto result_opt = fp_jit_opt.eval(params);

    EXPECT_NEAR((result_no_opt - result_opt).abs().max().template item<double>(), 0.0, epsilon);
    EXPECT_NEAR((result_opt - gold).abs().max().template item<double>(), 0.0, epsilon);

    if (!derivative_check)
      return;

    // Test derivatives using finite differences (commented out temporarily for debugging)
    std::vector<const torch::Tensor *> params_dx{&x2, &y, &n};
    std::vector<const torch::Tensor *> params_dy{&x, &y2, &n};
    std::vector<const torch::Tensor *> params_dn{&x, &y, &n2};

    auto derivativeTest = [&](const std::string & dvar, std::vector<const torch::Tensor *> dparams)
    {
      ParsedJITTensor dfp;
      if (dfp.parse(expression, variables))
      {
        const auto fp2 = fp_jit_opt.eval(dparams);
        const auto dr1 = (fp2 - result_opt) / eps_fd;

        dfp.differentiate(dvar);
        dfp.compile();
        const auto dr2 = dfp.eval(params);

        // Use relative error: |symbolic - fd| / (|fd| + eps) < tolerance
        const auto abs_diff = (dr2 - dr1).abs();
        const auto denominator = dr1.abs() + eps_fd;
        const auto rel_error = (abs_diff / denominator).max().template item<double>();

        // Relative tolerance: sqrt(epsilon) * 100 for first-order finite differences
        EXPECT_TRUE(rel_error < rel_tolerance ||
                    abs_diff.max().template item<double>() < abs_tolerance)
            << "Derivative w.r.t. " << dvar << "  failed for expression: " << expression << '\n'
            << "(" << rel_error << " < " << rel_tolerance << " || "
            << abs_diff.max().template item<double>() << " < " << abs_tolerance << ") is false";
      }
    };

    derivativeTest("x", params_dx);
    derivativeTest("y", params_dy);
    derivativeTest("n", params_dn);
  };

  auto check2 = [&](const std::string & expression,
                    std::string dvar,
                    const std::string & derivative,
                    bool compile)
  {
    std::vector<std::string> variables{"x", "y", "n"};
    std::vector<const torch::Tensor *> params{&x, &y, &n};

    ParsedJITTensor dF1, dF2;
    if (!dF1.parse(expression, variables))
      mooseError("Invalid JIT function: ", expression, "   ", dF1.errorMessage());
    if (!dF2.parse(derivative, variables))
      mooseError("Invalid JIT function: ", expression, "   ", dF2.errorMessage());

    dF1.differentiate(dvar);
    if (compile)
    {
      dF1.compile();
      dF2.compile();
    }

    const auto r1 = dF1.eval(params);
    const auto r2 = dF2.eval(params);

    const auto abs_diff = (r1 - r2).abs();
    const auto denominator = r1.abs() + epsilon;
    const auto rel_error = (abs_diff / denominator).max().template item<double>();

    EXPECT_NEAR(rel_error, 0.0, rel_tolerance);
  };

  // check hypot
  check("hypot(x,y)", torch::hypot(x, y));
  check("sqrt(x^2+y^2+n)", torch::sqrt(x * x + y * y + n));
  check("sqrt(x*x+y*y+n)", torch::sqrt(x * x + y * y + n));
  check("sqrt(x^2+y^2)", torch::sqrt(x * x + y * y));
  check("sqrt(x*x+y*y)", torch::sqrt(x * x + y * y));

  // trig functions
  check("tan((x-y)/2)", torch::tan((x - y) / 2.0));
  check("tanh(x-y)", torch::tanh(x - y));
  check("cos(y)", torch::cos(y));
  check("sin(y)", torch::sin(y));
  check("cosh(y)", torch::cosh(y));
  check("sinh(y)", torch::sinh(y));
  check("atan(x + y)", torch::atan(x + y));
  check("asin((x * y / 2) / n)", torch::asin((x * y / 2.0) / n));
  check("acos((x * y / 2) / n)", torch::acos((x * y / 2.0) / n));
  check("acosh(x+y+1)", torch::acosh(x + y + 1));
  check("asinh(x-y)", torch::asinh(x - y));
  check("atan2(x,y)", torch::atan2(x, y));

  // check reciprocal sqrt
  check("1/sqrt(x+y)", 1.0 / torch::sqrt(x + y));

  // check sin and cos together
  check("sin(y)-cos(y)", torch::sin(y) - torch::cos(y));

  // misc
  check("(x * y) / n", (x * y) / n);

  check("y/x", y / x, false);
  check2("y/x", "x", "-y/x^2", true);
  check2("y/x", "y", "1/x", true);

  check("-x", -x);
  check("rsqrt(x*y)", 1.0 / torch::sqrt(x * y));
  check("exp(x*y)", torch::exp(x * y));
  check("exp2(x*y)", torch::pow(2.0, x * y));
  check("(x*y) % 1.5", torch::remainder(x * y, 1.5));
  check("log(x)", torch::log(x));
  check("log10(x)", torch::log10(x));
  check("log2(x)", torch::log2(x));
  check("pow(y, x)", torch::pow(y, x));
  check("abs(y-x)", torch::abs(y - x), false);
  check("floor(x-y)", torch::floor(x - y), false);
  check("ceil(x-y)", torch::ceil(x - y), false);
  check("round(x-y)", torch::round(x - y), false);

  check("trunc(x-y)", torch::trunc(x - y));
  check("min(x^3,y^2)", torch::minimum(x * x * x, y * y));

  check("max(x^2,sin(4*y))", torch::maximum(x * x, torch::sin(y * 4.0)), false);
  check2("max(x^2,sin(4*y))", "x", "if(x^2>=sin(4*y),2*x,0)", true);
  check2("max(x^2,sin(4*y))", "y", "if(x^2>=sin(4*y),0,4*cos(4*y))", true);

  check("pow(2, x)", torch::pow(2, x));
  check("pow(x, 1.0/3.0)", torch::pow(x, 1.0 / 3.0));
  // check("cbrt(x)", torch::pow(x, 1.0 / 3.0));
  check("if(x<1 | y>=2, x, y)", torch::where(torch::logical_or(x < 1, y >= 2), x, y));
  check("if(x<=1 & y>2, x*x, 3*y)", torch::where(torch::logical_and(x <= 1, y > 2), x * x, y * 3));

  // local variables
  check("r2:=x^2+y^2; sqrt(r2)", torch::sqrt(x * x + y * y));
  check2("x2:=x^2; sinx2:=sin(x2); 4*sinx2", "x", "8*x*cos(x*x)", true);
  check2("a:=sin(x^2); a + 2*a + 3*a", "x", "12*x*cos(x^2)", true);
  check2("a:=sin(x^2); a + 2*a + 3*a", "x", "12*x*cos(x^2)", false);
}

TEST(ParsedTensorTest, Substitute)
{
  SwiftExpressionParser::Parser parser;

  // Test basic variable substitution
  {
    auto expr = parser.parse("x + y");
    ASSERT_TRUE(expr != nullptr);

    auto replacement = parser.parse("2*z");
    ASSERT_TRUE(replacement != nullptr);

    auto substituted = expr->substitute("x", replacement);
    EXPECT_EQ(substituted->toString(), "((2.000000 * z) + y)");
  }
  {
    auto expr = parser.parse("x * y");
    ASSERT_TRUE(expr != nullptr);

    auto replacement = parser.parse("2+z");
    ASSERT_TRUE(replacement != nullptr);

    auto substituted = expr->substitute("x", replacement);
    EXPECT_EQ(substituted->toString(), "((2.000000 + z) * y)");
  }

  // Test substitution in nested expressions
  {
    auto expr = parser.parse("sin(x) + cos(x) * x");
    ASSERT_TRUE(expr != nullptr);

    auto replacement = parser.parse("y^2");
    ASSERT_TRUE(replacement != nullptr);

    auto substituted = expr->substitute("x", replacement);
    EXPECT_EQ(substituted->toString(),
              "(sin((y ^ 2.000000)) + (cos((y ^ 2.000000)) * (y ^ 2.000000)))");
  }

  // Test substitution in let expressions without shadowing
  {
    auto expr = parser.parse("a := x + 1; a * x");
    ASSERT_TRUE(expr != nullptr);

    auto replacement = parser.parse("y + z");
    ASSERT_TRUE(replacement != nullptr);

    auto substituted = expr->substitute("x", replacement);
    // Should substitute x in both the binding and the body
    EXPECT_EQ(substituted->toString(), "a:=((y + z) + 1.000000); (a * (y + z))");
  }

  // Test substitution with variable shadowing
  {
    auto expr = parser.parse("a := x + 1; a * x");
    ASSERT_TRUE(expr != nullptr);

    auto replacement = parser.parse("y + z");
    ASSERT_TRUE(replacement != nullptr);

    auto substituted = expr->substitute("a", replacement);
    // Should NOT substitute 'a' anywhere because the binding defines 'a' (shadowing)
    EXPECT_EQ(substituted->toString(), "a:=(x + 1.000000); (a * x)");
  }

  // Test that substitution doesn't affect unrelated variables
  {
    auto expr = parser.parse("x + y + z");
    ASSERT_TRUE(expr != nullptr);

    auto replacement = parser.parse("42");
    ASSERT_TRUE(replacement != nullptr);

    auto substituted = expr->substitute("y", replacement);
    EXPECT_EQ(substituted->toString(), "((x + 42.000000) + z)");
  }

  // Test substitution in complex let expressions with multiple bindings
  {
    auto expr = parser.parse("a := x; b := a + 1; b * x");
    ASSERT_TRUE(expr != nullptr);

    auto replacement = parser.parse("2*z");
    ASSERT_TRUE(replacement != nullptr);

    auto substituted = expr->substitute("x", replacement);
    // x should be substituted in first binding and in body, but not in second binding (uses 'a')
    EXPECT_EQ(substituted->toString(),
              "a:=(2.000000 * z); b:=(a + 1.000000); (b * (2.000000 * z))");
  }

  // Test substitution preserves expression structure
  {
    auto expr = parser.parse("r := x^2 + y^2; sqrt(r) + r");
    ASSERT_TRUE(expr != nullptr);

    auto replacement = parser.parse("t + 1");
    ASSERT_TRUE(replacement != nullptr);

    auto substituted = expr->substitute("x", replacement);
    // x should be substituted but local variable r should still be referenced
    EXPECT_EQ(substituted->toString(),
              "r:=(((t + 1.000000) ^ 2.000000) + (y ^ 2.000000)); (sqrt(r) + r)");
  }
}

TEST(ParsedTensorTest, ErrorHandling)
{
  SwiftExpressionParser::Parser parser;

  // Test invalid syntax - missing operand
  {
    auto expr = parser.parse("x + ");
    EXPECT_EQ(expr, nullptr);
    EXPECT_EQ(parser.errorMessage(),
              "Line 1:5: syntax error, expecting <IDENTIFIER>, <NUMBER>, '('.");
  }

  // Test invalid syntax - unmatched parenthesis
  {
    auto expr = parser.parse("(x + y");
    EXPECT_EQ(expr, nullptr);
    EXPECT_EQ(parser.errorMessage(), "Line 1:7: syntax error, expecting ')'.");
  }

  // Test invalid syntax - unmatched closing parenthesis
  {
    auto expr = parser.parse("x + y)");
    EXPECT_EQ(expr, nullptr);
    EXPECT_EQ(parser.errorMessage(), "Line 1:6: syntax error, unexpected ')', expecting '|', '&'.");
  }

  // Test invalid function call - missing closing paren
  {
    auto expr = parser.parse("sin(x");
    EXPECT_EQ(expr, nullptr);
    EXPECT_EQ(parser.errorMessage(), "Line 1:6: syntax error, expecting ')'.");
  }

  // Test invalid let expression - missing assignment
  {
    auto expr = parser.parse("a := ; x + a");
    EXPECT_EQ(expr, nullptr);
    EXPECT_EQ(parser.errorMessage(),
              "Line 1:6: syntax error, unexpected ';', expecting <IDENTIFIER>, <NUMBER>, '('.");
  }

  // Test invalid operator sequence
  {
    auto expr = parser.parse("x + * y");
    EXPECT_EQ(expr, nullptr);
    EXPECT_EQ(parser.errorMessage(),
              "Line 1:5: syntax error, unexpected '*', expecting <IDENTIFIER>, <NUMBER>, '('.");
  }

  // Test empty expression
  {
    auto expr = parser.parse("");
    EXPECT_EQ(expr, nullptr);
    EXPECT_EQ(parser.errorMessage(),
              "Line 1:1: syntax error, expecting <IDENTIFIER>, <NUMBER>, '('.");
  }

  // Test invalid number format
  {
    auto expr = parser.parse("1.2.3 + x");
    EXPECT_EQ(expr, nullptr);
    EXPECT_EQ(parser.errorMessage(), "Line 1:4: syntax error, unexpected '.', expecting '|', '&'.");
  }

  // Test evaluation with undefined variables
  {
    ParsedJITTensor fp;
    std::vector<std::string> variables{"x"};
    ASSERT_TRUE(fp.parse("x + y", variables)); // Parse succeeds, y treated as variable

    // But buildGraph should fail because y is not in the variables list
    // This happens during compile/eval
    std::vector<const torch::Tensor *> params{};
    EXPECT_THROW(
        {
          try
          {
            fp.eval(params);
          }
          catch (const std::exception & e)
          {
            EXPECT_EQ(std::string(e.what()), "Parameter count mismatch");
            throw;
          }
        },
        std::exception);
  }

  // Test differentiation with respect to undefined variable should return 0
  {
    auto expr = parser.parse("x + y");
    ASSERT_TRUE(expr != nullptr);

    auto deriv = expr->differentiate("z"); // z doesn't appear in expression
    auto simplified = deriv->simplify();
    EXPECT_EQ(simplified->toString(), "0.000000");
  }
}

TEST(ParsedTensorTest, Simplify)
{
  SwiftExpressionParser::Parser parser;

  // Test constant folding - addition
  {
    auto expr = parser.parse("2 + 3");
    ASSERT_TRUE(expr != nullptr);

    auto simplified = expr->simplify();
    EXPECT_EQ(simplified->toString(), "5.000000");
  }

  // Test constant folding - multiplication
  {
    auto expr = parser.parse("4 * 5");
    ASSERT_TRUE(expr != nullptr);

    auto simplified = expr->simplify();
    EXPECT_EQ(simplified->toString(), "20.000000");
  }

  // Test constant folding - power
  {
    auto expr = parser.parse("2 ^ 3");
    ASSERT_TRUE(expr != nullptr);

    auto simplified = expr->simplify();
    EXPECT_EQ(simplified->toString(), "8.000000");
  }

  // Test algebraic simplification - multiply by zero
  {
    auto expr = parser.parse("x * 0");
    ASSERT_TRUE(expr != nullptr);

    auto simplified = expr->simplify();
    EXPECT_EQ(simplified->toString(), "0.000000");
  }

  // Test algebraic simplification - multiply by one
  {
    auto expr = parser.parse("x * 1");
    ASSERT_TRUE(expr != nullptr);

    auto simplified = expr->simplify();
    EXPECT_EQ(simplified->toString(), "x");
  }

  // Test algebraic simplification - add zero
  {
    auto expr = parser.parse("x + 0");
    ASSERT_TRUE(expr != nullptr);

    auto simplified = expr->simplify();
    EXPECT_EQ(simplified->toString(), "x");
  }

  // Test algebraic simplification - subtract zero
  {
    auto expr = parser.parse("x - 0");
    ASSERT_TRUE(expr != nullptr);

    auto simplified = expr->simplify();
    EXPECT_EQ(simplified->toString(), "x");
  }

  // Test algebraic simplification - divide by one
  {
    auto expr = parser.parse("x / 1");
    ASSERT_TRUE(expr != nullptr);

    auto simplified = expr->simplify();
    EXPECT_EQ(simplified->toString(), "x");
  }

  // Test algebraic simplification - power of zero
  {
    auto expr = parser.parse("x ^ 0");
    ASSERT_TRUE(expr != nullptr);

    auto simplified = expr->simplify();
    EXPECT_EQ(simplified->toString(), "1.000000");
  }

  // Test algebraic simplification - power of one
  {
    auto expr = parser.parse("x ^ 1");
    ASSERT_TRUE(expr != nullptr);

    auto simplified = expr->simplify();
    EXPECT_EQ(simplified->toString(), "x");
  }

  // Test function constant folding
  {
    auto expr = parser.parse("sin(0)");
    ASSERT_TRUE(expr != nullptr);

    auto simplified = expr->simplify();
    // sin(0) = 0
    auto c = std::dynamic_pointer_cast<SwiftExpressionParser::Constant>(simplified);
    ASSERT_TRUE(c != nullptr);
    EXPECT_NEAR(c->value(), 0.0, 1e-10);
  }

  // Test nested simplification
  {
    auto expr = parser.parse("(x + 0) * 1 + 0");
    ASSERT_TRUE(expr != nullptr);

    auto simplified = expr->simplify();
    EXPECT_EQ(simplified->toString(), "x");
  }

  // Test simplification in let expressions
  {
    auto expr = parser.parse("a := 2 + 3; a * x");
    ASSERT_TRUE(expr != nullptr);

    auto simplified = expr->simplify();
    EXPECT_EQ(simplified->toString(), "a:=5.000000; (a * x)");
  }

  // Test complex constant folding
  {
    auto expr = parser.parse("sqrt(4) + log(1) + exp(0)");
    ASSERT_TRUE(expr != nullptr);

    auto simplified = expr->simplify();
    // sqrt(4) = 2, log(1) = 0, exp(0) = 1  => 2 + 0 + 1 = 3
    EXPECT_EQ(simplified->toString(), "3.000000");
  }
}

TEST(ParsedTensorTest, SecondDerivatives)
{
  // Use linspace values that avoid zero to prevent singularities
  const auto x =
      torch::unsqueeze(torch::linspace(0.1, 2.01, 11, MooseTensor::floatTensorOptions()), 1);

  const Real eps_fd = MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-4 : 1e-2;
  const Real rel_tolerance =
      MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-3 : 5e-2;

  auto testSecondDerivative = [&](const std::string & expression, const std::string & var)
  {
    std::vector<std::string> variables{var};
    std::vector<const torch::Tensor *> params{&x};

    // Compute second derivative symbolically
    ParsedJITTensor fp;
    ASSERT_TRUE(fp.parse(expression, variables));
    fp.differentiate(var);
    fp.differentiate(var); // Second derivative
    fp.compile();
    const auto d2f_symbolic = fp.eval(params);

    // Compute second derivative numerically using central differences
    ParsedJITTensor fp_orig;
    ASSERT_TRUE(fp_orig.parse(expression, variables));
    fp_orig.compile();

    const auto x_plus = x + eps_fd;
    const auto x_minus = x - eps_fd;

    std::vector<const torch::Tensor *> params_plus{&x_plus};
    std::vector<const torch::Tensor *> params_minus{&x_minus};

    const auto f_plus = fp_orig.eval(params_plus);
    const auto f_minus = fp_orig.eval(params_minus);
    const auto f_center = fp_orig.eval(params);

    // Second derivative: (f(x+h) - 2*f(x) + f(x-h)) / h^2
    const auto d2f_numerical = (f_plus - 2 * f_center + f_minus) / (eps_fd * eps_fd);

    // Check relative error
    const auto abs_diff = (d2f_symbolic - d2f_numerical).abs();
    const auto denominator = d2f_numerical.abs() + eps_fd;
    const auto rel_error = (abs_diff / denominator).max().template item<double>();

    EXPECT_TRUE(rel_error < rel_tolerance)
        << "Second derivative of " << expression << " w.r.t. " << var << " failed\n"
        << "Relative error: " << rel_error << " >= " << rel_tolerance;
  };

  // Test second derivatives of various functions
  testSecondDerivative("x^2", "x");     // d²/dx²[x²] = 2
  testSecondDerivative("x^3", "x");     // d²/dx²[x³] = 6x
  testSecondDerivative("sin(x)", "x");  // d²/dx²[sin(x)] = -sin(x)
  testSecondDerivative("cos(x)", "x");  // d²/dx²[cos(x)] = -cos(x)
  testSecondDerivative("exp(x)", "x");  // d²/dx²[exp(x)] = exp(x)
  testSecondDerivative("log(x)", "x");  // d²/dx²[log(x)] = -1/x²
  testSecondDerivative("1/x", "x");     // d²/dx²[1/x] = 2/x³
  testSecondDerivative("sqrt(x)", "x"); // d²/dx²[sqrt(x)] = -1/(4x^(3/2))

  // Test second derivative with let expressions
  testSecondDerivative("a := x^2; a * x", "x"); // d²/dx²[x³] = 6x
}

TEST(ParsedTensorTest, Constants)
{
  // Test using constants in expressions
  const auto x =
      torch::unsqueeze(torch::linspace(0.1, 2.01, 11, MooseTensor::floatTensorOptions()), 1);
  const auto pi_tensor = torch::full_like(x, M_PI);
  const auto e_tensor = torch::full_like(x, M_E);

  const Real epsilon = MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-12 : 1e-6;

  // Test expression with pi constant
  {
    ParsedJITTensor fp;
    std::vector<std::string> variables{"x"};
    std::unordered_map<std::string, torch::Tensor> constants{{"pi", pi_tensor}};

    ASSERT_TRUE(fp.parse("x * pi", variables, constants));
    fp.compile();

    std::vector<const torch::Tensor *> params{&x};
    const auto result = fp.eval(params);

    EXPECT_NEAR((result - x * M_PI).abs().max().template item<double>(), 0.0, epsilon);
  }

  // Test expression with multiple constants
  {
    ParsedJITTensor fp;
    std::vector<std::string> variables{"x"};
    std::unordered_map<std::string, torch::Tensor> constants{{"pi", pi_tensor}, {"e", e_tensor}};

    ASSERT_TRUE(fp.parse("sin(x) + pi * e", variables, constants));
    fp.compile();

    std::vector<const torch::Tensor *> params{&x};
    const auto result = fp.eval(params);
    const auto expected = torch::sin(x) + M_PI * M_E;

    EXPECT_NEAR((result - expected).abs().max().template item<double>(), 0.0, epsilon);
  }

  // Test differentiation with constants (constants should have zero derivative)
  {
    ParsedJITTensor fp;
    std::vector<std::string> variables{"x"};
    std::unordered_map<std::string, torch::Tensor> constants{{"pi", pi_tensor}};

    ASSERT_TRUE(fp.parse("x * pi", variables, constants));
    fp.differentiate("x");
    fp.compile();

    std::vector<const torch::Tensor *> params{&x};
    const auto result = fp.eval(params);

    // d/dx[x * pi] = pi
    EXPECT_NEAR((result - pi_tensor).abs().max().template item<double>(), 0.0, epsilon);
  }

  // Test that constants don't get confused with variables
  {
    SwiftExpressionParser::Parser parser;
    std::unordered_set<std::string> constants{"pi"};

    auto expr = parser.parse("x + pi", constants);
    ASSERT_TRUE(expr != nullptr);

    // Differentiate w.r.t. x (pi is constant, so its derivative is 0)
    auto deriv = expr->differentiate("x");
    auto simplified = deriv->simplify();

    // d/dx[x + pi] = 1 + 0 = 1
    EXPECT_EQ(simplified->toString(), "1.000000");
  }

  // Test constant in let expression
  {
    ParsedJITTensor fp;
    std::vector<std::string> variables{"x"};
    std::unordered_map<std::string, torch::Tensor> constants{{"pi", pi_tensor}};

    ASSERT_TRUE(fp.parse("a := x * pi; a + a", variables, constants));
    fp.compile();

    std::vector<const torch::Tensor *> params{&x};
    const auto result = fp.eval(params);
    const auto expected = 2.0 * x * M_PI;

    EXPECT_NEAR((result - expected).abs().max().template item<double>(), 0.0, epsilon);
  }
}
