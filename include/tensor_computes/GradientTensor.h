/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

#ifdef NEML2_ENABLED
#include "neml2/tensors/Vec.h"
using GradientTensorType = neml2::Vec;
#else
using GradientTensorType = torch::Tensor;
#endif

/**
 * Gradient of a tensor field
 */
class GradientTensor : public TensorOperator<GradientTensorType>
{
public:
  static InputParameters validParams();

  GradientTensor(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _input;
  const bool _input_is_reciprocal;

  const torch::Tensor _zero;
};
