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
 * DeGeus mechanics test object
 */
class MacroscopicShearTensor : public TensorOperator<>
{
public:
  static InputParameters validParams();

  MacroscopicShearTensor(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _tF;
};
