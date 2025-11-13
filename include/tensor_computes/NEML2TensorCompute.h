/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
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

protected:
#ifdef NEML2_ENABLED
  std::shared_ptr<neml2::Model> _model;

  std::vector<std::pair<const torch::Tensor *, neml2::LabeledAxisAccessor>> _input_mapping;
  std::vector<std::pair<neml2::LabeledAxisAccessor, torch::Tensor *>> _output_mapping;
#endif
};
