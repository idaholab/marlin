/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperatorBase.h"

#ifdef NEML2_ENABLED
#include "NEML2ModelHandle.h"
#endif

/**
 * NEML2 compute object
 */
class NEML2TensorCompute : public TensorOperatorBase
{
public:
  static InputParameters validParams();

  NEML2TensorCompute(const InputParameters & parameters);

  void init() override;
  void computeBuffer() override;

  /// NEML2 models have their own JIT tracing infrastructure
  virtual bool supportsJIT() const override { return false; }

protected:
#ifdef NEML2_ENABLED
  std::unique_ptr<NEML2ModelHandle> _model;

  std::vector<std::pair<const torch::Tensor *, std::string>> _input_mapping;
  std::vector<std::pair<std::string, torch::Tensor *>> _output_mapping;
#endif
};
