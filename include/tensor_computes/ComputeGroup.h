/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperatorBase.h"

#include <memory>

class JITExecutor;

/**
 * Compute group with internal dependency resolution and optional JIT tracing
 */
class ComputeGroup : public TensorOperatorBase
{
public:
  static InputParameters validParams();

  ComputeGroup(const InputParameters & parameters);

  virtual void init() override;

  virtual void computeBuffer() override;

  virtual void updateDependencies() override;

  /// Called when grid changes - invalidates JIT caches
  virtual void gridChanged() override;

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

  /// JIT execution manager
  std::unique_ptr<JITExecutor> _jit_executor;

  /// Whether JIT tracing is enabled for this group
  const bool _jit_enabled;
};
