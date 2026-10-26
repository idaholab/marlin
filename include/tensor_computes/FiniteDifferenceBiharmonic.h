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
 * factor * Laplacian(Laplacian(u)) with the discrete Laplacian of
 * FiniteDifferenceLaplacian applied twice (the 13-point stencil in 2D for the
 * 3-point Laplacian).
 */
class FiniteDifferenceBiharmonic : public TensorOperator<torch::Tensor>, public LinearFDOperator
{
public:
  static InputParameters validParams();

  FiniteDifferenceBiharmonic(const InputParameters & parameters);

  virtual void init() override;

  virtual void realSpaceComputeBuffer() override;

  virtual void computeBuffer() override;

  virtual std::vector<FDStencilEntry> stencilAt() const override;
  virtual std::vector<FDStencilEntry>
  stencilAt(const std::vector<Real> & grid_spacing) const override;
  virtual Definiteness definiteness() const override;
  virtual const std::string & inputBufferName() const override { return _input_name; }

  /// the Laplacian stencil composed with itself, times factor (one entry per offset)
  static std::vector<FDStencilEntry>
  buildStencil(unsigned int dim, bool wide, const std::vector<Real> & grid_spacing, Real factor);

protected:
  /// stencil radius of the underlying Laplacian
  const unsigned int _laplacian_radius;
  /// input buffer name
  const TensorInputBufferName _input_name;
  /// input buffer
  const torch::Tensor & _u_in;
  /// prefactor
  const Real _factor;
  /// convolution kernel of the stencil
  torch::Tensor _kernel;
  /// per face, the sign of the reflection that sets the ghosts of w = Laplacian(u) on this rank's
  /// domain faces: -1 for a Dirichlet face (Laplacian(u) = 0), +1 for a Neumann face
  /// (dLaplacian(u)/dn = 0), 0 for none (periodic, a neighbouring rank, or another condition)
  std::vector<int> _w_reflection;
  /// apply the Laplacian twice with the w ghosts above (3-point stencil with such faces only)
  bool _two_stage;
};
