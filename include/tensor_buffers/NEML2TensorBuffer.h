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
template <typename T>
class NEML2TensorBuffer : public TensorBuffer<T>
{
public:
  static InputParameters validParams();

  NEML2TensorBuffer(const InputParameters & parameters);

  virtual void init() override;
  virtual void makeCPUCopy() override;

  using TensorBuffer<T>::_u;
  using TensorBuffer<T>::_u_cpu;
  using TensorBuffer<T>::_cpu_copy_requested;
};

#ifdef NEML2_ENABLED
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/SR2.h"

using VectorTensor = NEML2TensorBuffer<neml2::Vec>;
registerTensorType(VectorTensor, neml2::Vec);

using SR2Tensor = NEML2TensorBuffer<neml2::SR2>;
registerTensorType(SR2Tensor, neml2::SR2);

#else

// placeholder class
using VectorTensor = NEML2TensorBuffer<torch::Tensor>;
using SR2Tensor = NEML2TensorBuffer<torch::Tensor>;

#endif
