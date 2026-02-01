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
 * Linear operator for the two-mode FCC phase-field crystal model.
 */
class FCCPFCLinear : public TensorOperator<>
{
public:
  static InputParameters validParams();

  FCCPFCLinear(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const Real _eps;
  const Real _q1;
  const Real _r1;
  const Real _mobility;
  const torch::Tensor & _k2;
};
