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
 * Tensor wrapper for NEML2-coupled tensor buffers.
 *
 * NEML2 v3 exchanges plain torch tensors, so this buffer stores a torch::Tensor
 * just like the other Marlin tensor buffers.
 */
class NEML2TensorBuffer : public TensorBuffer<torch::Tensor>
{
public:
  static InputParameters validParams();

  NEML2TensorBuffer(const InputParameters & parameters);

  virtual void init() override;
  virtual void makeCPUCopy() override;
};

using VectorTensor = NEML2TensorBuffer;
using SR2Tensor = NEML2TensorBuffer;
