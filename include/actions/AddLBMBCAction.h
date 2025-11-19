/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseObjectAction.h"

/**
 * This class adds LBM BC object
 */

class AddLBMBCAction : public MooseObjectAction
{
public:
  static InputParameters validParams();

  AddLBMBCAction(const InputParameters & parameters);
  virtual void act() override;
};
