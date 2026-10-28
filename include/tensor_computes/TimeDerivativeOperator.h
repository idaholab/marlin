/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/
#pragma once

#include "TensorOperator.h"
#include "LinearFDOperator.h"

/**
 * Time derivative term for implicit solves, backward Euler or variable-step BDF2. With the
 * weights w0, w1, w2 of the step the action is w0 u/dt and the right-hand side contribution is
 * -(w1 u^n + w2 u^{n-1})/dt, so that together with the spatial operators the solver solves
 * (w0 u + w1 u^n + w2 u^{n-1})/dt + sum_i L_i u = f. Backward Euler is w0 = 1, w1 = -1, w2 = 0.
 */
class TimeDerivativeOperator : public TensorOperator<torch::Tensor>, public LinearFDOperator
{
public:
  static InputParameters validParams();

  TimeDerivativeOperator(const InputParameters & parameters);

  virtual void realSpaceComputeBuffer() override;

  virtual void computeBuffer() override;

  virtual std::vector<FDStencilEntry> stencilAt() const override;
  virtual std::vector<FDStencilEntry>
  stencilAt(const std::vector<Real> & grid_spacing) const override;
  virtual Definiteness definiteness() const override { return Definiteness::POSITIVE_DEFINITE; }
  virtual const std::string & inputBufferName() const override { return _input_name; }

  /**
   * -(w1 u^n + w2 u^{n-1})/dt on the owned cells; call once per solve attempt, before the solve
   * overwrites the input buffer. It also sets the weights of the step (from its time, a retried
   * step keeps the history) used by the action and the stencil.
   */
  virtual torch::Tensor rhsContribution() const override;

protected:
  /// input buffer name
  const TensorInputBufferName _input_name;
  /// input buffer
  const torch::Tensor & _u_in;
  /// input buffer wrapper (for the owned-cell view)
  TensorBufferBase & _u_in_base;

  /// 1 (backward Euler) or 2 (BDF2)
  const unsigned int _order;

  /// weight of u in the current step (1 for backward Euler and the first BDF2 step)
  mutable Real _w0;

  /// BDF2 history: u^n and u^{n-1} on the owned cells, the start time of the step that starts
  /// from u^n, and the dt of the last solve attempt and of the accepted step ending at u^n
  mutable torch::Tensor _u_n;
  mutable torch::Tensor _u_n_minus_1;
  mutable Real _t_n;
  mutable Real _last_dt;
  mutable Real _dt_old;
};
