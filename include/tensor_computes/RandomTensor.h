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
 * Random IC
 */
class RandomTensor : public TensorOperator<>
{
public:
  static InputParameters validParams();

  RandomTensor(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const bool _generate_on_cpu;
};
