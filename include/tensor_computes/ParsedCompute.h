/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"
#include "ParsedJITTensor.h"
#include <array>

/**
 * ParsedCompute - JIT-compiled mathematical expression evaluator
 *
 * This tensor operator parses mathematical expressions and evaluates them
 * using a JIT-compiled PyTorch compute graph for optimal performance.
 * Supports symbolic differentiation, algebraic simplification, and
 * all standard mathematical operations.
 */
class ParsedCompute : public TensorOperator<>
{
public:
  static InputParameters validParams();

  ParsedCompute(const InputParameters & parameters);

  void computeBuffer() override;
  void realSpaceComputeBuffer() override;

  /// Block this for now, because time is not properly updated
  virtual bool supportsJIT() const override { return false; }

protected:
  const bool _extra_symbols;

  ParsedJITTensor _parser;

  torch::Tensor _time_tensor;
  std::vector<const torch::Tensor *> _params;
  std::array<const torch::Tensor *, 3> _axis_params;
  std::array<torch::Tensor, 3> _axis_padded;
  enum class ExpandEnum
  {
    REAL,
    RECIPROCAL,
    NONE
  } _expand;

  const bool _is_integer;
};
