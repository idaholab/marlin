/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * Create a constant 3x3 random rotation matrix tensor over the domain.
 */
class RandomRotationTensor : public TensorOperator<>
{
public:
  static InputParameters validParams();
  RandomRotationTensor(const InputParameters & parameters);

  void computeBuffer() override;
  bool supportsJIT() const override { return false; }

protected:
  const bool _generate_on_cpu;
  const bool _has_seed;
  const int _seed;
};
