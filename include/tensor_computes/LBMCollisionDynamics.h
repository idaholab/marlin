/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannOperator.h"

/**
 * Template object for LBM collision dynamics
 */
template <int coll_dyn>
class LBMCollisionDynamicsTempl : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMCollisionDynamicsTempl(const InputParameters & parameters);

  void HermiteRegularization();
  void computeRelaxationParameter();
  void computeLocalRelaxationMatrix();
  void computeGlobalRelaxationMatrix();

  void BGKDynamics();
  void MRTDynamics();
  void SmagorinskyDynamics();
  void SmagorinskyMRTDynamics();

  void computeBuffer() override;

protected:
  const torch::Tensor & _f;
  const torch::Tensor & _feq;
  /// reference to externally provided relaxation matrix [Nx, Ny, Nz, Q] input buffer
  const torch::Tensor & _input_relaxation_matrix;
  /// reference to externally provided relaxation tensor [Nx, Ny, Nz] input buffer
  const torch::Tensor & _tau_tensor;
  /// non-equilibrium distribution function
  torch::Tensor _fneq;
  /// local shear stress relaxation tau_s
  torch::Tensor _local_relaxation_parameter;
  /// local relaxation matrix [Nx, Ny, Nz, Q] (internally computed for Smagorinsky)
  torch::Tensor _local_relaxation_matrix;
  /// global relaxation matrix [Q]
  torch::Tensor _global_relaxation_matrix;

  /// Precomputed Matrices for Hermite Projection
  /// Lattice velocity outer products [Q, 9]
  torch::Tensor _C_mat;
  /// Scaled Hermite basis projection operator [9, Q]
  torch::Tensor _P_mat;
  /// Pre-allocated flat buffer for non-equilibrium stress tensor [N, 9]
  torch::Tensor _pi_neq_flat;

  /// MRT Specific Caches
  /// Precomputed static MRT collision matrix [Q, Q]
  torch::Tensor _MSM_t;
  /// Flat buffer for MRT moment space [N, Q]
  torch::Tensor _m_neq_flat;

  /// Smagorinsky Specific Caches
  /// Flattened outer products of lattice velocities [Q, 27]
  torch::Tensor _outer_flat;

  std::vector<int64_t> _shape_with_ghost;

  const Real _tau_0;
  const Real _C_s;     // Smagorinsky constant
  const Real _delta_x; // grid resolution
  const bool _projection;
  const bool _is_dynamic_relaxation;
  Real _mean_density;
};

typedef LBMCollisionDynamicsTempl<0> LBMBGKCollision;
typedef LBMCollisionDynamicsTempl<1> LBMMRTCollision;
typedef LBMCollisionDynamicsTempl<2> LBMSmagorinskyCollision;
typedef LBMCollisionDynamicsTempl<3> LBMSmagorinskyMRTCollision;
