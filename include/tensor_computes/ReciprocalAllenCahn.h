/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * ReciprocalAllenCahn (\nabla M \nabla \mu) object
 */
class ReciprocalAllenCahn : public TensorOperator<>
{
public:
  static InputParameters validParams();

  ReciprocalAllenCahn(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _dF_chem_deta;
  const torch::Tensor & _L;
  /// imaginary unit i
  const torch::Tensor _imag;
  const torch::Tensor & _psi;
  bool _update_psi;
  const bool _always_update_psi;
  torch::Tensor _psi_thresh;
};
