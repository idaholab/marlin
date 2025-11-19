/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannOperator.h"

/**
 *  Stack given scalar tensor buffers and output vectorial tensor
 */
class LBMStackTensors : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMStackTensors(const InputParameters &);

  virtual void init() override;
  void computeBuffer() override;

protected:
  const std::vector<TensorInputBufferName> & _buffer_names;
};
