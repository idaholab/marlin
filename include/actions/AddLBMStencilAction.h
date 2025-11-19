/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseObjectAction.h"

/**
 * This class adds LBM stencil object.
 */

class AddLBMStencilAction : public MooseObjectAction
{
public:
  static InputParameters validParams();

  AddLBMStencilAction(const InputParameters & parameters);
  virtual void act() override;
};
