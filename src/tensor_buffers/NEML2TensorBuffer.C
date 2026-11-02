/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "NEML2TensorBuffer.h"

registerMooseObject("MarlinApp", VectorTensor);
registerMooseObject("MarlinApp", SR2Tensor);

InputParameters
NEML2TensorBuffer::validParams()
{
  InputParameters params = TensorBuffer<torch::Tensor>::validParams();
  params.addClassDescription("This tensor type requires Marlin to be built with NEML2.");
  return params;
}

NEML2TensorBuffer::NEML2TensorBuffer(const InputParameters & parameters)
  : TensorBuffer<torch::Tensor>(parameters)
{
#ifndef NEML2_ENABLED
  mooseError("This tensor type requires Marlin to be built with NEML2.");
#else
  mooseInfoRepeated("Instantiating NEML2 tensor class");
#endif
}

void
NEML2TensorBuffer::init()
{
  // No NEML2-specific initialization; buffer storage is managed by Marlin.
}

void
NEML2TensorBuffer::makeCPUCopy()
{
#ifndef NEML2_ENABLED
  mooseError("This tensor type requires Marlin to be built with NEML2.");
#else
  if (!_u.defined())
    return;

  if (_cpu_copy_requested)
  {
    if (_u.is_cpu())
      _u_cpu = _u.clone().contiguous();
    else
      _u_cpu = _u.cpu().contiguous();
  }
#endif
}
