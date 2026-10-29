/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TimeDerivativeOperator.h"

/**
 * Time derivative term a du/dt with a cell-centred coefficient a >= 0 (e.g. the domain parameter
 * psi of the smoothed boundary method), backward Euler or variable-step BDF2 as
 * TimeDerivativeOperator. The action is a w0 u/dt and the right-hand side contribution is
 * -a (w1 u^n + w2 u^{n-1})/dt, with a the coefficient when the step's right-hand side is formed.
 */
class CoefficientTimeDerivativeOperator : public TimeDerivativeOperator
{
public:
  static InputParameters validParams();

  CoefficientTimeDerivativeOperator(const InputParameters & parameters);

  virtual void computeBuffer() override;

  virtual std::vector<FDStencilEntry>
  stencilAt(const std::vector<Real> & grid_spacing) const override;
  virtual std::vector<std::vector<int>> stencilOffsets() const override;

  /// -a (w1 u^n + w2 u^{n-1})/dt on the owned cells (see TimeDerivativeOperator)
  virtual torch::Tensor rhsContribution() const override;

  /// a on this rank's owned block
  virtual torch::Tensor ownedCoefficient() const override;

  /// the centre entry a w0/dt from a with one halo layer
  virtual std::vector<FDStencilEntry>
  stencilFromCoefficient(const torch::Tensor & coefficient,
                         const std::vector<Real> & grid_spacing) const override;

protected:
  /// a on the owned cells, checked to be nonnegative
  torch::Tensor checkedCoefficient() const;

  /// coefficient buffer name
  const TensorInputBufferName _coefficient_name;
  /// coefficient buffer wrapper (for the owned-cell view)
  TensorBufferBase & _coefficient_base;

  /// the owned a of the step's right-hand side, which the action and the stencil must use
  mutable torch::Tensor _step_coefficient;
};
