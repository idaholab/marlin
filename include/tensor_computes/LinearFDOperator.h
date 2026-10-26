/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseTypes.h"
#include <torch/torch.h>
#include <string>
#include <vector>

/**
 * One term of a finite-difference stencil: a grid-point offset (one entry
 * per spatial dimension) and the coefficient it multiplies. Coefficients
 * already include any 1/dx^2-style grid-spacing scaling and operator
 * prefactor, matching exactly what the operator's matrix-free action
 * applies. A variable-coefficient operator also gives a per-cell field, which
 * multiplies the coefficient in each row (the row's cell, not the column's).
 */
struct FDStencilEntry
{
  std::vector<int> offset;
  Real coefficient;
  /// per-cell factor on this rank's owned block of the grid the stencil is for (Marlin
  /// axis order); undefined for a constant coefficient
  torch::Tensor field;
};

/**
 * Mixin interface for TensorComputes whose action is a linear, known
 * finite-difference stencil (e.g. FiniteDifferenceLaplacian). This is NOT a
 * TensorOperatorBase subclass; it is multiply-inherited alongside
 * TensorOperator<T> by operators that can expose their stencil for uses
 * beyond matrix-free evaluation (e.g. assembling a preconditioner matrix).
 */
class LinearFDOperator
{
public:
  virtual ~LinearFDOperator() = default;

  /**
   * Returns this operator's finite-difference stencil as grid-point offsets
   * and coefficients, matching the coefficients used by the operator's own
   * matrix-free (e.g. convolution-based) action exactly.
   */
  virtual std::vector<FDStencilEntry> stencilAt() const = 0;

  /**
   * The same stencil rediscretised on a grid with the given spacing (one entry
   * per dimension). Implicit solvers use it to build coarse multigrid levels.
   */
  virtual std::vector<FDStencilEntry> stencilAt(const std::vector<Real> & grid_spacing) const = 0;

  /// Definiteness of the operator's action A (as written, before any sign change)
  enum class Definiteness
  {
    POSITIVE_DEFINITE,
    /// symmetric positive semidefinite, nullspace = constants (e.g. -Laplacian, periodic)
    POSITIVE_SEMIDEFINITE,
    /// neither of the above (indefinite or non-symmetric)
    OTHER
  };

  virtual Definiteness definiteness() const = 0;

  /// Name of the buffer the operator acts on
  virtual const std::string & inputBufferName() const = 0;

  /**
   * Contribution r of this operator to the right-hand side of the implicit
   * system sum_i A_i u = sum_i r_i + f, evaluated on the owned cells before the
   * solve. Undefined (the default) means no contribution.
   */
  virtual torch::Tensor rhsContribution() const { return {}; }
};
