# SwiftExpressionParser

## Overview

SwiftExpressionParser is a custom mathematical expression parser built on [cpp-peglib](https://github.com/yhirose/cpp-peglib) that provides robust parsing, symbolic differentiation, algebraic simplification, and JIT compilation of mathematical expressions for Swift. It replaces the previous FParser-based implementation with a cleaner, more maintainable architecture based on Abstract Syntax Trees (ASTs).

## Architecture

### Design Philosophy

The parser follows a clean separation of concerns with distinct phases:

1. **Parsing**: PEG grammar → AST construction
2. **Transformation**: Symbolic differentiation and algebraic simplification
3. **Compilation**: AST → PyTorch JIT compute graph
4. **Execution**: JIT-optimized tensor operations

This separation enables powerful features like:
- Symbolic differentiation at the AST level
- Algebraic simplification before graph construction
- Multiple optimization passes (AST + torch JIT)
- Easy extensibility for new operations

### Core Components

#### Abstract Syntax Tree (AST)

The parser constructs an AST using a polymorphic node hierarchy:

```cpp
class Expr {
  virtual ExprPtr simplify() const = 0;
  virtual ExprPtr differentiate(const std::string & var) const = 0;
  virtual torch::jit::Value * buildGraph(...) const = 0;
  virtual std::string toString() const = 0;
};
```

**Node Types:**

- **`Constant`**: Numeric literals (e.g., `3.14`, `1.5e-3`)
- **`Variable`**: Input variables from buffers or tensors
- **`ConstantTensor`**: Named constants (e.g., `pi`, `e`, `i`)
- **`BinaryOp`**: Binary operations (`+`, `-`, `*`, `/`, `^`)
- **`UnaryOp`**: Unary operations (`-`, `!`)
- **`Comparison`**: Comparisons (`<`, `>`, `<=`, `>=`, `==`, `!=`)
- **`LogicalOp`**: Logical operations (`&`, `|`)
- **`FunctionCall`**: Built-in functions (e.g., `sin`, `cos`, `sqrt`)
- **`LetExpression`**: Local variable bindings with `:=` operator

#### PEG Grammar

The parser uses a Parsing Expression Grammar with operator precedence:

```peg
EXPRESSION  <- STATEMENTS
STATEMENTS  <- (ASSIGNMENT ';')* LOGICAL
ASSIGNMENT  <- IDENTIFIER ':=' LOGICAL
LOGICAL     <- COMPARISON (LOGICAL_OP COMPARISON)*
COMPARISON  <- ADDITIVE (COMP_OP ADDITIVE)?
ADDITIVE    <- MULTITIVE (ADD_OP MULTITIVE)*
MULTITIVE   <- UNARY (MUL_OP UNARY)*
UNARY       <- (UNARY_OP UNARY) / POWER
POWER       <- PRIMARY ('^' POWER)?
PRIMARY     <- FUNCTION / NUMBER / VARIABLE / '(' LOGICAL ')'
```

**Operator Precedence** (highest to lowest):
1. Function calls, parentheses
2. Power (`^`) - right-associative
3. Unary operators (`-`, `!`)
4. Multiplication, division (`*`, `/`)
5. Addition, subtraction (`+`, `-`)
6. Comparisons (`<`, `>`, `<=`, `>=`, `==`, `!=`)
7. Logical AND (`&`)
8. Logical OR (`|`)
9. Local assignments (`:=`)

## Features

### Supported Operations

#### Binary Operators

- **Arithmetic**: `+`, `-`, `*`, `/`, `^` (power)
- **Comparison**: `<`, `>`, `<=`, `>=`, `==`, `!=`
- **Logical**: `&` (and), `|` (or)

#### Unary Operators

- **Negation**: `-x`
- **Logical NOT**: `!condition`

#### Mathematical Functions

- **Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- **Hyperbolic**: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- **Exponential/Logarithmic**: `exp`, `log`, `log10`, `log2`
- **Power/Root**: `pow`, `sqrt`, `cbrt`
- **Rounding**: `floor`, `ceil`, `round`, `trunc`
- **Other**: `abs`, `min`, `max`, `erf`, `erfc`, `if`

#### Special Functions

- **`if(condition, true_value, false_value)`**: Ternary conditional that maps to `torch::where`

### Local Variables

Local variables enable subexpression reuse and improved code clarity using the `:=` assignment operator:

```
r0 := 10;
r := sqrt(x^2 + y^2);
tanh(r - r0)
```

**Features:**
- Variables are evaluated in order (left-to-right)
- Later assignments can reference earlier ones
- Variables are immutable (functional style)
- Full differentiation support via chain rule
- Local scope (don't pollute outer environment)

**Example:**
```cpp
// Smooth interface disk
expression = 'r0:=10; r:=sqrt(x^2+y^2); tanh(r-r0)'

// Multiple intermediate calculations
expression = 'a:=x+y; b:=x*y; c:=sqrt(a^2+b^2); c/10'
```

### Constants Map

The parser accepts a constants map for predefined mathematical and physical constants:

```cpp
std::unordered_map<std::string, torch::Tensor> constants;
constants["pi"] = torch::tensor(M_PI);
constants["e"] = torch::tensor(M_E);
constants["i"] = torch::tensor(c10::complex<double>(0.0, 1.0));
constants["kB"] = torch::tensor(8.617e-5);  // Boltzmann constant
```

Constants are treated as tensor inputs during graph construction, enabling:
- Proper torch broadcasting
- Type safety
- Easy addition of new constants
- No string substitution required

### Symbolic Differentiation

The parser performs symbolic differentiation at the AST level using standard calculus rules:

#### Differentiation Rules

**Constants and Variables:**
```cpp
d/dx(c) = 0              // Constant
d/dx(x) = 1              // Variable w.r.t. itself
d/dx(y) = 0              // Variable w.r.t. different variable
```

**Binary Operations:**
```cpp
d/dx(f + g) = df/dx + dg/dx                    // Sum rule
d/dx(f - g) = df/dx - dg/dx                    // Difference rule
d/dx(f * g) = df/dx * g + f * dg/dx            // Product rule
d/dx(f / g) = (df/dx * g - f * dg/dx) / g^2    // Quotient rule
d/dx(f ^ g) = f^g * (dg/dx * ln(f) + g * df/dx / f)  // General power rule
d/dx(f ^ c) = c * f^(c-1) * df/dx              // Power rule (constant exponent)
```

**Function Derivatives:**
```cpp
d/dx(sin(f)) = cos(f) * df/dx
d/dx(cos(f)) = -sin(f) * df/dx
d/dx(exp(f)) = exp(f) * df/dx
d/dx(log(f)) = df/dx / f
d/dx(sqrt(f)) = df/dx / (2 * sqrt(f))
// ... and more
```

**Local Variables:**

For `let v := e in body`, differentiation uses the chain rule:
```cpp
d/dx(let v := e in body) =
  let v := e, dv := de/dx in
    d(body)/dx
```

The body differentiation treats local variables as implicitly dependent on the outer variable.

### Algebraic Simplification

The `simplify()` method performs algebraic transformations to optimize expressions:

#### Constant Folding

Evaluate constant expressions at compile time:
```cpp
2 + 3 → 5
sin(0) → 0
2 * pi → 6.283...
```

#### Algebraic Identities

**Addition:**
```cpp
x + 0 → x
0 + x → x
```

**Multiplication:**
```cpp
x * 0 → 0
0 * x → 0
x * 1 → x
1 * x → x
```

**Division:**
```cpp
x / 1 → x
0 / x → 0
```

**Power:**
```cpp
x ^ 0 → 1
x ^ 1 → x
```

**Negation:**
```cpp
-(-x) → x
-(0) → 0
```

### JIT Compilation

After AST simplification, the parser converts the AST to a PyTorch JIT compute graph:

#### Graph Construction

Each AST node implements `buildGraph()` to emit corresponding torch operations:

```cpp
torch::jit::Value * BinaryOp::buildGraph(
    torch::jit::Graph & graph,
    std::unordered_map<std::string, torch::jit::Value *> & vars) const
{
  auto left = _left->buildGraph(graph, vars);
  auto right = _right->buildGraph(graph, vars);

  switch (_op) {
    case Op::Add: return graph.insert(torch::jit::aten::add, {left, right});
    case Op::Mul: return graph.insert(torch::jit::aten::mul, {left, right});
    // ...
  }
}
```

#### Torch Optimizations

After graph construction, multiple torch JIT optimization passes are applied:

1. **EliminateDeadCode**: Remove unused computations
2. **ConstantPropagation**: Fold constants in the graph
3. **EliminateCommonSubexpression**: Deduplicate identical subgraphs
4. **FuseGraph**: Fuse element-wise operations into single kernels

#### Execution

The optimized graph is executed via `GraphExecutor`:

```cpp
torch::jit::Stack stack;
for (auto & param : params)
  stack.push_back(*param);
for (auto & constant : constants)
  stack.push_back(constant.second);

executor->run(stack);
return stack[0].toTensor();
```

## Usage

### Basic Parsing

```cpp
#include "ParsedJITTensor.h"

ParsedJITTensor parser;
std::vector<std::string> variables{"x", "y"};

// Parse expression
if (!parser.parse("x^2 + 2*x*y + y^2", variables)) {
  std::cerr << "Error: " << parser.errorMessage() << std::endl;
  return;
}

// Compile for optimal performance
parser.compile();

// Evaluate
auto x = torch::tensor(2.0);
auto y = torch::tensor(3.0);
std::vector<const torch::Tensor*> params{&x, &y};
auto result = parser.eval(params);  // → 25.0
```

### With Constants

```cpp
std::unordered_map<std::string, torch::Tensor> constants;
constants["pi"] = torch::tensor(M_PI);
constants["e"] = torch::tensor(M_E);

parser.parse("2*pi*r", variables, constants);
```

### With Differentiation

```cpp
parser.parse("r^2", variables);
parser.differentiate("r");  // Computes d/dr(r^2) = 2*r
parser.compile();
auto result = parser.eval(params);  // Returns 2*r
```

### With Local Variables

```cpp
parser.parse("r:=sqrt(x^2+y^2); r^2", variables);
parser.differentiate("x");  // Chain rule applied automatically
```

## Integration with ParsedCompute

[ParsedCompute](ParsedCompute.md) uses SwiftExpressionParser for expression evaluation in tensor computations:

```
[compute]
  type = ParsedCompute
  buffer = result
  expression = 'r0:=10; r:=sqrt(x^2+y^2); tanh(r-r0)'
  derivatives = 'x y'  # Optional: compute derivatives
  inputs = ''           # Optional: buffer inputs
  extra_symbols = true  # Enable x,y,z,t,pi,e,i symbols
[]
```

### Parameters

- **`expression`**: Mathematical expression string
- **`inputs`**: List of input buffer names (become variables)
- **`derivatives`**: List of variables to differentiate with respect to
- **`extra_symbols`**: Add spatial coordinates (x,y,z), wavenumbers (kx,ky,kz), k², time (t), and constants (pi, e, i)
- **`constant_names`**: User-defined constant names
- **`constant_expressions`**: Expressions for user-defined constants

## Performance Considerations

### Two-Stage Optimization

1. **AST Level**: Constant folding, algebraic simplification
2. **Graph Level**: Torch JIT optimizations (fusion, CSE, etc.)

This two-stage approach provides:
- Early elimination of unnecessary work
- Better optimization opportunities for torch JIT
- Cleaner intermediate representations

### Compilation vs. Interpretation

Always call `compile()` before repeated evaluations:

```cpp
parser.parse(expr, vars);
parser.compile();  // JIT-compile once

// Fast repeated evaluations
for (int i = 0; i < 1000; i++)
  auto result = parser.eval(params);
```

Without `compile()`, the parser builds a new graph on each `eval()`, which is significantly slower.

### Memory Management

- AST nodes use `shared_ptr` for automatic memory management
- Compiled graphs are cached in the parser instance
- No manual cleanup required

## Testing

See [test/tests/parsed_tensor/local_vars_derivative.i](/test/tests/parsed_tensor/local_vars_derivative.i) for a comprehensive test that verifies:

- Local variable assignment
- Symbolic differentiation with local variables
- Comparison against hand-coded derivatives
- Integration with tensor postprocessors

## Future Enhancements

Potential improvements:

1. **Advanced optimizations**:
   - Strength reduction (x*x → x^2)
   - Loop hoisting for repeated subexpressions
   - Common subexpression extraction at AST level

2. **Additional operations**:
   - Matrix operations (dot, cross, norm)
   - Tensor slicing and indexing
   - Piecewise functions

3. **Performance profiling**:
   - Expression complexity metrics
   - Optimization effectiveness reporting

4. **Better error messages**:
   - Syntax error location and context
   - Type mismatch detection
   - Undefined variable warnings

## References

- [cpp-peglib](https://github.com/yhirose/cpp-peglib): PEG parser library
- [PyTorch JIT](https://pytorch.org/docs/stable/jit.html): Just-in-time compilation
- [ParsedCompute](ParsedCompute.md): User-facing tensor compute object
- [ParsedJITTensor](../../include/utils/ParsedJITTensor.h): Parser implementation
