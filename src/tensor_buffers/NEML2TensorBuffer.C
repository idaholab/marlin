/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "NEML2TensorBuffer.h"

registerMooseObject("MarlinApp", VectorTensor);
registerMooseObject("MarlinApp", SR2Tensor);

template <typename T>
InputParameters
NEML2TensorBuffer<T>::validParams()
{
  InputParameters params = TensorBuffer<T>::validParams();
  if constexpr (std::is_same_v<T, torch::Tensor>)
    params.addClassDescription("This tensor type requires Marlin to be built woth NEML2.");
  return params;
}

template <typename T>
NEML2TensorBuffer<T>::NEML2TensorBuffer(const InputParameters & parameters)
  : TensorBuffer<T>(parameters)
{
  if constexpr (std::is_same_v<T, torch::Tensor>)
    mooseError("This tensor type requires Marlin to be built woth NEML2.");

  mooseInfoRepeated("Instantiating NEML2 tensor class ", libMesh::demangle(typeid(T).name()));
}

template <typename T>
void
NEML2TensorBuffer<T>::init()
{
  // TODO
}

template <typename T>
void
NEML2TensorBuffer<T>::makeCPUCopy()
{
  if constexpr (std::is_same_v<T, torch::Tensor>)
    mooseError("This tensor type requires Marlin to be built woth NEML2.");
  else
  {
    if (!_u.defined())
      return;

    if (_cpu_copy_requested)
    {
      if (_u.is_cpu())
        _u_cpu = T(_u.clone().contiguous(), _u.batch_dim());
      else
        _u_cpu = T(_u.cpu().contiguous(), _u.batch_dim());
    }
  }
}

#ifdef NEML2_ENABLED
template class NEML2TensorBuffer<neml2::Vec>;
template class NEML2TensorBuffer<neml2::SR2>;
#else
template class NEML2TensorBuffer<torch::Tensor>;
#endif
