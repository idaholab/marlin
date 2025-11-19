/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannStencilBase.h"

/**
 * 3-dimensional 19 velocity lattice configuration
 */

class LBMD3Q19 : public LatticeBoltzmannStencilBase
{
public:
  static InputParameters validParams();

  LBMD3Q19(const InputParameters & parameters);
};
