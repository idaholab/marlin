/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperatorBase.h"
#include <ATen/core/TensorBody.h>

/**
 * TensorSolver object (this is mostly a compute object)
 */
class TensorSolver : public TensorOperatorBase
{
public:
  static InputParameters validParams();

  TensorSolver(const InputParameters & parameters);

  virtual void computeBuffer() override;

  virtual void updateDependencies() override final;

  /// Solvers have iterative algorithms with data-dependent control flow
  virtual bool supportsJIT() const override { return false; }

protected:
  const std::vector<torch::Tensor> & getBufferOld(const std::string & param,
                                                  unsigned int max_states,
                                                  unsigned int ghost_layers = 0);
  const std::vector<torch::Tensor> & getBufferOldByName(const TensorInputBufferName & buffer_name,
                                                        unsigned int max_states,
                                                        unsigned int ghost_layers = 0);

  void gatherDependencies();
  void forwardBuffers();

  /// perform the actual solver substep
  virtual void substep() = 0;

  /// Number of substeps per time step
  const unsigned int _substeps;

  /// current substep number
  unsigned int _substep;

  /// references to the substep dt/time managed by the TensorProblem
  Real & _sub_dt;
  Real & _sub_time;

  /// MOOSE timestep
  const Real & _dt;
  const Real & _dt_old;

  /// root compute for the solver
  std::shared_ptr<TensorOperatorBase> _compute;

  /// forwarded buffers
  std::vector<std::pair<torch::Tensor &, const torch::Tensor &>> _forwarded_buffers;
};
