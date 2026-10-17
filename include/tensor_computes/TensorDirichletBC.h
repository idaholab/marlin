/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorRobinBC.h"

/// Dirichlet preset of TensorRobinBC: a=1, b=0, c=value (u = value on the face).
class TensorDirichletBC : public TensorRobinBC
{
public:
  static InputParameters validParams();

  TensorDirichletBC(const InputParameters & parameters);
};
