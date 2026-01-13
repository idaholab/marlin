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
  torch::Tensor _fneq;
  torch::Tensor _relaxation_parameter;
  torch::Tensor _local_relaxation_matrix;
  torch::Tensor _global_relaxation_matrix;

  const Real _tau_0;
  const Real _C_s;     // Smagorinsky constant
  const Real _delta_x; // grid resolution
  const bool _projection;
  Real _mean_density;
};

typedef LBMCollisionDynamicsTempl<0> LBMBGKCollision;
typedef LBMCollisionDynamicsTempl<1> LBMMRTCollision;
typedef LBMCollisionDynamicsTempl<2> LBMSmagorinskyCollision;
typedef LBMCollisionDynamicsTempl<3> LBMSmagorinskyMRTCollision;
