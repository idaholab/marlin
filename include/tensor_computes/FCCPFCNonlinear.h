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
 * Nonlinear operator for the two-mode FCC phase-field crystal model.
 */
class FCCPFCNonlinear : public TensorOperator<>
{
public:
  static InputParameters validParams();

  FCCPFCNonlinear(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _psi;
  const torch::Tensor * _dealiasing;
  const Real _mobility;
  const torch::Tensor & _k2;
};
