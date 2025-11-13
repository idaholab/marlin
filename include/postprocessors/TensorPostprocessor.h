/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "GeneralPostprocessor.h"
#include "GeneralVectorPostprocessor.h"
#include "DomainInterface.h"
#include <torch/torch.h>

class TensorProblem;

/**
 * Postprocessor that operates on a buffer
 */
template <class T>
class TensorPostprocessorTempl : public T, public DomainInterface
{
public:
  static InputParameters validParams();

  TensorPostprocessorTempl(const InputParameters & parameters);

protected:
  TensorProblem & _tensor_problem;

  /// The buffer this postprocessor is operating on
  const torch::Tensor & _u;
};

typedef TensorPostprocessorTempl<GeneralPostprocessor> TensorPostprocessor;
typedef TensorPostprocessorTempl<GeneralVectorPostprocessor> TensorVectorPostprocessor;
