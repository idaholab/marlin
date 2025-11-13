/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * Swift-Hohenberg linear operator
 */
class SwiftHohenbergLinear : public TensorOperator<>
{
public:
  static InputParameters validParams();

  SwiftHohenbergLinear(const InputParameters & parameters);

  virtual void computeBuffer() override;

  const Real _r;
  const Real _alpha;
  const torch::Tensor & _k2;
};
