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
#include "neml2/base/LabeledAxisAccessor.h"
#include "neml2/models/Model.h"
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
  std::shared_ptr<neml2::Model> _model;

  std::vector<std::tuple<const torch::Tensor *, neml2::TensorType, neml2::LabeledAxisAccessor>>
      _input_mapping;
  std::vector<std::tuple<const std::vector<torch::Tensor> *,
                         const torch::Tensor *,
                         neml2::TensorType,
                         neml2::LabeledAxisAccessor>>
      _old_input_mapping;

  std::vector<std::pair<neml2::LabeledAxisAccessor, torch::Tensor *>> _output_mapping;
#endif
};
