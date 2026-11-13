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
  /// the input field, without the trailing unit dimension of 2D buffers
  torch::Tensor scalarField() const;
  torch::Tensor padScalarField(const torch::Tensor & field) const;
  torch::Tensor prepareInputField();
  torch::Tensor stencilConvolve3D(const torch::Tensor & field,
                                  const torch::Tensor & kernel_flat) const;

  const torch::Tensor & _scalar_field;
  const int64_t _padding = 1;
  bool _is_interior = false;

  torch::Tensor _kernel;

  /// (ex, ey, ez) of every stencil direction in kernel order (3D only)
  std::vector<std::array<int64_t, 3>> _offsets;
  torch::nn::functional::Conv2dFuncOptions _conv2d_options;
};
