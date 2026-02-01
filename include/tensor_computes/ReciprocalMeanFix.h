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
 * Fix the k=0 mode in reciprocal space to enforce a target mean value.
 */
class ReciprocalMeanFix : public TensorOperator<>
{
public:
  static InputParameters validParams();
  ReciprocalMeanFix(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _input;
  const Real _u_inf;
};
