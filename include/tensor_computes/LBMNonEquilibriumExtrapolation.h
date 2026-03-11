/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LBMBoundaryCondition.h"

/**
 * Non-equilibrium extrapolation boundary condition (Guo et al., 2002)
 */
class LBMNonEquilibriumExtrapolation : public LBMBoundaryCondition
{
public:
  static InputParameters validParams();

  LBMNonEquilibriumExtrapolation(const InputParameters & parameters);

  void topBoundary() override;
  void bottomBoundary() override;
  void leftBoundary() override;
  void rightBoundary() override;
  void frontBoundary() override;
  void backBoundary() override;
  void computeBuffer() override;

protected:
  torch::Tensor computeEquilibriumSlice(const torch::Tensor & rho_b,
                                        const torch::Tensor & ux_b,
                                        const torch::Tensor & uy_b,
                                        const torch::Tensor & uz_b) const;
  torch::Tensor computeDensitySlice(const torch::Tensor & f_slice) const;
  void computeVelocitySlice(const torch::Tensor & f_slice,
                            const torch::Tensor & rho_slice,
                            torch::Tensor & ux,
                            torch::Tensor & uy,
                            torch::Tensor & uz) const;
  void applyNEE(int dim, int64_t b_idx, int64_t n1_idx, int64_t n2_idx);

  const MooseEnum _prescribe_type;
  const int _order;
  const Real _prescribed_ux;
  const Real _prescribed_uy;
  const Real _prescribed_uz;
  const Real _prescribed_rho;
};
