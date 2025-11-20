/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include <ATen/core/TensorBody.h>
#include "TensorOperator.h"

/**
 * Constant Tensor
 */
class FFTMechanics : public TensorOperator<>
{
public:
  static InputParameters validParams();

  FFTMechanics(const InputParameters & parameters);

  virtual void check() override;
  virtual void computeBuffer() override;

protected:
  // // stiffness
  // neml2::SSR4 _C;

  /// R2 identity tensor
  const torch::Tensor _ti;
  const torch::Tensor _tI;
  const torch::Tensor _tI4;
  const torch::Tensor _tI4rt;
  const torch::Tensor _tI4s;
  const torch::Tensor _tII;

  /// Material parameters
  const torch::Tensor & _tK;
  const torch::Tensor & _tmu;

  const std::vector<int64_t> _r2_shape;

  /// current deformation gradient
  const torch::Tensor & _tF;

  /// Gamma projection operator
  torch::Tensor _Ghat4;

  /// stress
  const torch::Tensor & _tP;
  /// tangent operator
  const torch::Tensor & _tK4;

  Real _l_tol;
  unsigned int _l_max_its;
  Real _nl_rel_tol;
  Real _nl_abs_tol;
  unsigned int _nl_max_its;

  TensorOperatorBase & _constitutive_model;

  /// applied macroscopic (affine) strain
  const torch::Tensor * const _applied_macroscopic_strain;

  const bool _verbose;

  using TensorOperatorBase::_dim;
  using TensorOperatorBase::_domain;
};
