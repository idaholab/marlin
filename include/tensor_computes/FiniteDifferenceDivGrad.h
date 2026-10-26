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

#include <utility>

/**
 * factor * div(D grad u) with a cell-centred coefficient D, as the difference of the face
 * fluxes D_face (u_j - u_i) / h on each axis. D_face is the harmonic mean of the two cells'
 * D (the default) or their arithmetic mean. On a domain face D_face is the boundary cell's
 * own D. The stencil is the 3-point Laplacian's, with one coefficient field per offset.
 */
class FiniteDifferenceDivGrad : public TensorOperator<torch::Tensor>, public LinearFDOperator
{
public:
  static InputParameters validParams();

  FiniteDifferenceDivGrad(const InputParameters & parameters);

  virtual void realSpaceComputeBuffer() override;

  virtual void computeBuffer() override;

  virtual std::vector<FDStencilEntry> stencilAt() const override;
  virtual std::vector<FDStencilEntry>
  stencilAt(const std::vector<Real> & grid_spacing) const override;
  virtual std::vector<std::vector<int>> stencilOffsets() const override;
  virtual Definiteness definiteness() const override;
  virtual const std::string & inputBufferName() const override { return _input_name; }

protected:
  /// the owned block of a tensor with the halo, shifted by shift cells along axis
  torch::Tensor ownedBlock(const torch::Tensor & t, unsigned int axis = 0, int64_t shift = 0) const;

  /// D's halo: from the neighbouring ranks and periodic images, D_0 on a domain face
  void exchangeCoefficient() const;

  /// D on the lower and upper faces of each owned cell along axis (exchangeCoefficient first)
  std::pair<torch::Tensor, torch::Tensor> faceCoefficients(unsigned int axis) const;

  /// input buffer name
  const TensorInputBufferName _input_name;
  /// input buffer
  const torch::Tensor & _u_in;
  /// coefficient buffer name
  const TensorInputBufferName _coefficient_name;
  /// coefficient buffer D
  const torch::Tensor & _coefficient;
  /// prefactor
  const Real _factor;
  /// harmonic (true) or arithmetic face mean
  const bool _harmonic;
};
