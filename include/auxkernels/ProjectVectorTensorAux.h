/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "AuxKernel.h"
#include "TensorProblemInterface.h"
#include "DomainInterface.h"
#include "torch/torch.h"

#if 0

class TensorProblem;

/**
 * Map a vectorial TensorBuffer to an AuxVariable
 * In the future this can be combined with ProjectTensorAux
 */
class ProjectVectorTensorAux : public ArrayAuxKernel,
                               public TensorProblemInterface,
                               public DomainInterface
{
public:
  static InputParameters validParams();

  ProjectVectorTensorAux(const InputParameters & parameters);

protected:
  virtual RealEigenVector computeValue() override;

  const torch::Tensor & _cpu_buffer;

  const unsigned int & _dim;
  const std::array<int64_t, 3> & _n;
  const RealVectorValue & _grid_spacing;
};

#endif
