/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorBuffer.h"

/**
 * Tensor wrapper arbitrary tensor value dimensions
 */
class PlainTensorBuffer : public TensorBuffer<torch::Tensor>
{
public:
  static InputParameters validParams();

  PlainTensorBuffer(const InputParameters & parameters);

  virtual void init() override;
  virtual void makeCPUCopy() override;
};

registerTensorType(PlainTensorBuffer, torch::Tensor);
