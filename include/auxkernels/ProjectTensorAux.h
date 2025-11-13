/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "AuxKernel.h"
#include "TensorProblemInterface.h"
#include "DomainInterface.h"
#include <torch/torch.h>

class TensorProblem;

/**
 * Map an TensorBuffer to an AuxVariable
 */
class ProjectTensorAux : public AuxKernel, public TensorProblemInterface, public DomainInterface
{
public:
  static InputParameters validParams();

  ProjectTensorAux(const InputParameters & parameters);

protected:
  virtual Real computeValue() override;

  const torch::Tensor & _cpu_buffer;

  const unsigned int & _dim;
  const std::array<int64_t, 3> & _n;
  const RealVectorValue & _grid_spacing;
};
