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
 * Identity rank two tensor Tensor
 */
class RankTwoIdentity : public TensorOperator<>
{
public:
  static InputParameters validParams();

  RankTwoIdentity(const InputParameters & parameters);

  virtual void computeBuffer() override;

  /// mesh dimension
  const unsigned int & _dim;
};
