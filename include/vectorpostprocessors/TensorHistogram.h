/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorPostprocessor.h"

/**
 * Compute the integral of a Tensor buffer
 */
class TensorHistogram : public TensorVectorPostprocessor
{
public:
  static InputParameters validParams();

  TensorHistogram(const InputParameters & parameters);

  virtual void initialize() override {}
  virtual void execute() override;
  virtual void finalize() override {}

protected:
  const Real _min;
  const Real _max;
  std::size_t _bins;
  torch::Tensor _bin_edges;

  VectorPostprocessorValue & _bin_vec;
  VectorPostprocessorValue & _count_vec;
};
