/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannOperator.h"

/**
 * Compute gradient with isotropic discretization scheme
 */
class LBMIsotropicGradient : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMIsotropicGradient(const InputParameters & parameters);

  torch::Tensor padScalarField();
  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _scalar_field;
  const int64_t _padding = (3 - 1) / 2;
  torch::IntArrayRef _pad_dims = {_padding, _padding, _padding, _padding};

  torch::Tensor _kernel;
  torch::nn::functional::Conv2dFuncOptions _conv_options;
};
