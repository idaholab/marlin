/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorPredictor.h"

/**
 * Linear TensorPredictor object
 */
class LinearTensorPredictor : public TensorPredictor
{
public:
  static InputParameters validParams();

  LinearTensorPredictor(const InputParameters & parameters);

  /// perform the computation
  virtual void computeBuffer();

protected:
  const Real _scale;
};
