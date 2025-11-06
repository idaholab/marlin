/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ParsedCompute.h"

#include "MooseUtils.h"
#include "SwiftUtils.h"
#include "MultiMooseEnum.h"
#include "DomainAction.h"
#include "libmesh/fparser_ad.hh"

registerMooseObject("SwiftApp", ParsedCompute);

InputParameters
ParsedCompute::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("ParsedCompute object.");
  params.addRequiredParam<std::string>("expression", "Parsed expression");
  params.addParam<std::vector<TensorInputBufferName>>(
      "inputs", {}, "Buffer names used in the expression");
  params.addParam<std::vector<TensorInputBufferName>>(
      "derivatives", {}, "List of inputs to take the derivative w.r.t. (or none)");
  params.addParam<bool>("enable_fpoptimizer", true, "Use algebraic optimizer");
  params.addParam<bool>("extra_symbols",
                        false,
                        "Provide i (imaginary unit), kx,ky,kz (reciprocal space frequency), k2 "
                        "(square of the k-vector), x,y,z "
                        "(real space coordinates), time t, pi, and e.");
  // Constants and their values
  params.addParam<std::vector<std::string>>(
      "constant_names",
      std::vector<std::string>(),
      "Vector of constants used in the parsed function (use this for kB etc.)");
  params.addParam<std::vector<std::string>>(
      "constant_expressions",
      std::vector<std::string>(),
      "Vector of values for the constants in constant_names (can be an FParser expression)");
  MooseEnum expandEnum("REAL RECIPROCAL NONE", "NONE");
  params.addParam<MooseEnum>("expand", expandEnum, "Expand the tensor to full size.");
  params.addParam<bool>("is_integer", false, "Turn the function result into an integer tensor");
  return params;
}

ParsedCompute::ParsedCompute(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _extra_symbols(getParam<bool>("extra_symbols")),
    _expand(getParam<MooseEnum>("expand").getEnum<ExpandEnum>()),
    _is_integer(getParam<bool>("is_integer"))
{
  const auto & expression = getParam<std::string>("expression");
  const auto & names = getParam<std::vector<TensorInputBufferName>>("inputs");

  // check for duplicates
  auto hasDuplicates = [](const std::vector<std::string> & values)
  {
    std::set<std::string> s(values.begin(), values.end());
    return values.size() != s.size();
  };

  if (hasDuplicates(names))
    paramError("inputs", "Duplicate buffer name.");

  // get all input buffers
  for (const auto & name : names)
    _params.push_back(&getInputBufferByName(name));

  static const std::vector<std::string> reserved_symbols = {
      "i", "x", "kx", "y", "ky", "z", "kz", "k2", "t", "pi", "e"};

  // helper function to check if the name given is one of the reserved_names
  auto isReservedName = [this](const auto & name)
  { return _extra_symbols && std::count(reserved_symbols.begin(), reserved_symbols.end(), name); };

  const auto & constant_names = getParam<std::vector<std::string>>("constant_names");
  const auto & constant_expressions = getParam<std::vector<std::string>>("constant_expressions");

  if (hasDuplicates(constant_names))
    paramError("constant_names", "Duplicate constant name.");

  for (const auto & name : constant_names)
    if (isReservedName(name))
      paramError("constant_names", "Cannot use reserved name '", name, "' for constant.");
  for (const auto & name : names)
    if (isReservedName(name))
      paramError("inputs", "Cannot use reserved name '", name, "' for coupled fields.");

  // check constant vectors
  unsigned int nconst = constant_expressions.size();
  if (nconst != constant_names.size())
    paramError("constant_names",
               "The parameter vectors constant_names (size ",
               constant_names.size(),
               ") and constant_values (size ",
               nconst,
               ") must have equal length.");

  // Evaluate user-provided constants
  std::vector<Real> constant_values(nconst);
  for (unsigned int i = 0; i < nconst; ++i)
  {
    // no need to use dual numbers for the constant expressions
    auto expr = std::make_shared<FunctionParserADBase<Real>>();

    // add previously evaluated constants
    for (unsigned int j = 0; j < i; ++j)
      if (!expr->AddConstant(constant_names[j], constant_values[j]))
        paramError("constant_names", "Invalid constant name '", constant_names[j], "'");

    // build the temporary constant expression function
    if (expr->Parse(constant_expressions[i], "") >= 0)
      mooseError("Invalid constant expression\n",
                 constant_expressions[i],
                 "\n in parsed function object.\n",
                 expr->ErrorMsg());

    constant_values[i] = expr->Eval(nullptr);
  }

  // Build list of variables
  std::vector<std::string> variables_vec = names;

  // Build constants map for mathematical constants and user constants
  std::unordered_map<std::string, torch::Tensor> constants_map;

  // Add user-provided constants as scalar tensors
  for (unsigned int i = 0; i < nconst; ++i)
    constants_map[constant_names[i]] =
        torch::tensor(constant_values[i], MooseTensor::floatTensorOptions());

  // Add extra symbols if requested
  if (_extra_symbols)
  {
    // Add mathematical constants (pi, e, i) to the constants map
    constants_map["pi"] = torch::tensor(libMesh::pi, MooseTensor::floatTensorOptions());
    constants_map["e"] = torch::tensor(std::exp(Real(1.0)), MooseTensor::floatTensorOptions());
    constants_map["i"] =
        torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions());

    // Add tensor variables (x, kx, y, ky, z, kz, k2, t) to variables list
    static const std::vector<std::string> tensor_symbols = {
        "x", "kx", "y", "ky", "z", "kz", "k2", "t"};
    variables_vec.insert(variables_vec.end(), tensor_symbols.begin(), tensor_symbols.end());

    // Add tensor variable parameters
    for (const auto dim : make_range(3u))
    {
      _params.push_back(&_domain.getAxis(dim));
      _params.push_back(&_domain.getReciprocalAxis(dim));
    }
    _params.push_back(&_domain.getKSquare());
    _params.push_back(&_time_tensor);
  }

  // Parse the expression with variables and constants
  if (!_parser.parse(expression, variables_vec, constants_map))
    paramError("expression", "Invalid function: ", _parser.errorMessage());

  // Take derivatives
  for (const auto & d : getParam<std::vector<TensorInputBufferName>>("derivatives"))
  {
    if (std::find(names.begin(), names.end(), d) != names.end())
      _parser.differentiate(d);
    else
      paramError("derivatives",
                 "Derivative w.r.t `",
                 d,
                 "` was requested, but it is not listed in `inputs`.");
  }

  // Compile (optimization is done during compilation if enable_fpoptimizer is true)
  // The compile step performs AST simplification and JIT graph optimization
  _parser.compile();
}

void
ParsedCompute::computeBuffer()
{
  if (_extra_symbols)
    _time_tensor = torch::tensor(_time, MooseTensor::floatTensorOptions());

  // Evaluate using JIT-compiled graph
  _u = _parser.eval(_params);

  if (_is_integer)
    _u = _u.to(MooseTensor::intTensorOptions());

  // optionally expand the tensor
  switch (_expand)
  {
    case ExpandEnum::REAL:
      _u = _u.expand(_domain.getShape());
      break;

    case ExpandEnum::RECIPROCAL:
      _u = _u.expand(_domain.getReciprocalShape());
      break;

    case ExpandEnum::NONE:
      break;
  }
}
