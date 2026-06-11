/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
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

  virtual void computeBuffer() override;

protected:
  torch::Tensor padScalarField();
  torch::Tensor prepareInputField();
  torch::Tensor stencilConvolve3D(const torch::Tensor & field,
                                  const torch::Tensor & kernel_flat) const;

  const torch::Tensor & _scalar_field;
  const int64_t _padding = 1;
  bool _is_interior = false;

  torch::Tensor _kernel;
  torch::nn::functional::Conv2dFuncOptions _conv2d_options;
};
