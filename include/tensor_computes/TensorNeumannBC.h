/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorRobinBC.h"

/// Neumann preset of TensorRobinBC: a=0, b=1, c=flux (du/dn = flux on the face).
class TensorNeumannBC : public TensorRobinBC
{
public:
  static InputParameters validParams();

  TensorNeumannBC(const InputParameters & parameters);
};
