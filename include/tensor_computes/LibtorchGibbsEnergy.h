/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"
// moose headers
#include "DataFileUtils.h"
// libtorch headers
#include <torch/script.h>

class LibtorchGibbsEnergy : public TensorOperator<>
{
public:
  static InputParameters validParams();

  LibtorchGibbsEnergy(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  unsigned int _n_phases;
  std::vector<const torch::Tensor *> _phase_fractions;
  std::vector<torch::Tensor *> _domega_detas;

  unsigned int _n_components;
  std::vector<const torch::Tensor *> _concentrations;
  std::vector<torch::Tensor *> _chemical_potentials;

  Moose::DataFileUtils::Path _file_path;
  // We need to use a pointer here because forward is not const qualified
  std::unique_ptr<torch::jit::script::Module> _surrogate;
};
