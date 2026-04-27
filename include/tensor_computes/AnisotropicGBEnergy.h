/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "DataFileUtils.h"
#include "TensorOperator.h"

#include <torch/script.h>

/**
 * Evaluate anisotropic grain-boundary energy from a TorchScript model.
 */
class AnisotropicGBEnergy : public TensorOperator<>
{
public:
  static InputParameters validParams();

  AnisotropicGBEnergy(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  double epsilonFor(const torch::ScalarType dtype) const;

  const torch::Tensor & _gb_normal_buffer;
  torch::Tensor & _dsigma_dn;
  Moose::DataFileUtils::Path _file_path;

  // forward() is not const-qualified
  std::unique_ptr<torch::jit::script::Module> _surrogate;
};
