/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannOperator.h"

#if 0
/**
 * LBMTensorUnitConverter object
 */
class LBMTensorUnitConverter : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMTensorUnitConverter(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _tensor_buffer;
  const Real & _conversion_constant;
};

#endif
