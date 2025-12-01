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
 * Computes the Laplacian using finite differences with ghost layer support.
 */
class FiniteDifferenceLaplacian : public TensorOperator<torch::Tensor>
{
public:
  static InputParameters validParams();

  FiniteDifferenceLaplacian(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const unsigned int _radius;
  /// input buffer
  const torch::Tensor & _u_in;
};
