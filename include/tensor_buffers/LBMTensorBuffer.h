/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorBuffer.h"

class LatticeBoltzmannStencilBase;
class LatticeBoltzmannProblem;

/**
 * Tensor wrapper for LBM tensors
 */
class LBMTensorBuffer : public TensorBuffer<torch::Tensor>
{
public:
  static InputParameters validParams();

  LBMTensorBuffer(const InputParameters & parameters);

  void init() override;
  virtual void makeCPUCopy() override;

  void readTensorFromFile(const std::vector<int64_t> &);
  void readTensorFromHdf5();

protected:
  const std::string _buffer_type;
  LatticeBoltzmannProblem & _lb_problem;
  const LatticeBoltzmannStencilBase & _stencil;
};
