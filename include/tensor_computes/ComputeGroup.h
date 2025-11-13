/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperatorBase.h"

/**
 * Compute group with internal dependency resolution
 */
class ComputeGroup : public TensorOperatorBase
{
public:
  static InputParameters validParams();

  ComputeGroup(const InputParameters & parameters);

  virtual void init() override;

  virtual void computeBuffer() override;

  virtual void updateDependencies() override;

  std::size_t getComputeCount() const { return _compute_count; }

protected:
  /// nested tensor computes
  std::vector<std::shared_ptr<TensorOperatorBase>> _computes;

  /// for diagnostic purposes we can make sure that every requested buffer is defined
  typedef std::vector<std::tuple<const torch::Tensor *, std::string, std::string>>
      CheckedTensorList;
  std::vector<CheckedTensorList> _checked_tensors;

  bool _visited;

  std::size_t _compute_count;
};
