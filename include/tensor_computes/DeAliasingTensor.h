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
 * De-aliasing filter
 */
class DeAliasingTensor : public TensorOperator<>
{
public:
  static InputParameters validParams();

  DeAliasingTensor(const InputParameters & parameters);

  virtual void computeBuffer() override;

  const enum class DeAliasingMethod { SHARP, HOULI } _method;

  const Real _p;
  const Real _alpha;
};
