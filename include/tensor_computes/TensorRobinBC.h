/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorBoundaryCondition.h"

/**
 * General Robin boundary condition: a*u + b*du/dn = c on the given face,
 * where n is the outward normal. Both the explicit ghost-fill path
 * (fillGhostCells) and the implicit matrix-row/RHS path (stencilAt, from
 * LinearFDOperator, unused until a later PETSc PR) are derived from the same
 * (a, b, c) triple, by construction, so the two cannot drift apart.
 */
class TensorRobinBC : public TensorBoundaryCondition
{
public:
  static InputParameters validParams();

  TensorRobinBC(const InputParameters & parameters);

  virtual void fillGhostCells(torch::Tensor & buffer,
                              unsigned int ghost_layers,
                              int64_t halo,
                              int64_t owned_extent,
                              Real dx) const override;

  virtual std::pair<std::vector<FDStencilEntry>, Real>
  stencilContribution(unsigned int ghost_layers, Real dx) const override;

  /**
   * Solve a*ubar + b*(u0-ughost)/dx_signed = c for ughost, where ubar is the
   * boundary-face value approximation (u0+ughost)/2 and dx_signed carries the
   * outward-normal sign convention (+dx on the upper face, -dx on the lower
   * face, consistent between fillGhostCells and stencilContribution). This is
   * the single source of truth both paths build from, analogous to
   * FiniteDifferenceLaplacian::buildStencil(): no MOOSE/TensorProblem
   * dependency, so it is directly unit-testable, and the required R4
   * dual-path consistency test calls fillGhostCellsStatic() and
   * stencilContributionStatic() below (the actual production logic, not a
   * re-derivation of it) on the same (a, b, c, dx) to confirm they imply the
   * same boundary value.
   */
  static Real ghostCoefficientOnU0(Real a, Real b, Real dx_signed);
  static Real ghostConstantTerm(Real a, Real b, Real c, Real dx_signed);

  /// Object-independent core of fillGhostCells(), used by both the instance method and the R4 test.
  static void fillGhostCellsStatic(torch::Tensor & buffer,
                                   bool is_upper_face,
                                   int64_t face_dim,
                                   unsigned int ghost_layers,
                                   int64_t halo,
                                   int64_t owned_extent,
                                   Real a,
                                   Real b,
                                   Real c,
                                   Real dx);

  /// Object-independent core of stencilContribution(), used by both the instance method and the R4 test.
  static std::pair<std::vector<FDStencilEntry>, Real>
  stencilContributionStatic(unsigned int dim, Real a, Real b, Real c, Real dx);

protected:
  /// Robin coefficients: a*u + b*du/dn = c
  const Real _a;
  const Real _b;
  const Real _c;
};
