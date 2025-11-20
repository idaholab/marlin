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
 * Compute LB checmial potential for pahse field coupling
 */
class LBMComputeChemicalPotential : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMComputeChemicalPotential(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _phi;
  const torch::Tensor & _laplacian_phi;

  const Real & _D;     // interface thickness
  const Real & _sigma; // interfacial tension coefficient
};
