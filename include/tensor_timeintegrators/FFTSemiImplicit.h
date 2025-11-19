/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorTimeIntegrator.h"

/**
 * TensorTimeIntegrator object (this is mostly a compute object)
 */
class FFTSemiImplicit : public TensorTimeIntegrator<>
{
public:
  static InputParameters validParams();

  FFTSemiImplicit(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const unsigned int _history_size;
  const torch::Tensor & _reciprocal_buffer;
  const torch::Tensor & _linear_reciprocal;
  const torch::Tensor & _non_linear_reciprocal;
  const std::vector<torch::Tensor> & _old_reciprocal_buffer;
  const std::vector<torch::Tensor> & _old_non_linear_reciprocal;
};
