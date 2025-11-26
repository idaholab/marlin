/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"
#include "FunctionInterface.h"

class Function;

/**
 * Constant Tensor
 */
class MooseFunctionTensor : public TensorOperator<>, public FunctionInterface
{
public:
  static InputParameters validParams();

  MooseFunctionTensor(const InputParameters & parameters);

  virtual void computeBuffer() override;

  /// MOOSE functions are not traceable by PyTorch JIT
  virtual bool supportsJIT() const override { return false; }

  const Function & _func;
};
