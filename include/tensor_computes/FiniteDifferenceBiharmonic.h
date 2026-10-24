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
};
