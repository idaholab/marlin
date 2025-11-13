/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

class SmoothRectangleCompute : public TensorOperator<>
{
public:
  static InputParameters validParams();

  SmoothRectangleCompute(const InputParameters & parameters);

  virtual void computeBuffer() override;

  const Real _x1;
  const Real _x2;
  const Real _y1;
  const Real _y2;
  const Real _z1;
  const Real _z2;

  const enum class interpolationFunction { COS, TANH } _interp_func;
  const Real _int_width;

  const Real _inside;
  const Real _outside;
};
