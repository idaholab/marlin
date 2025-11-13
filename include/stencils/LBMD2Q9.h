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
 * 2-dimensional 9 velocity lattice configuration
 */

class LBMD2Q9 : public LatticeBoltzmannStencilBase
{
public:
  static InputParameters validParams();

  LBMD2Q9(const InputParameters & parameters);
};
