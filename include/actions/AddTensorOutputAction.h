/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseObjectAction.h"

class TensorProblem;

/**
 * This class adds an TensorOutput object.
 */
class AddTensorOutputAction : public MooseObjectAction
{
public:
  static InputParameters validParams();

  AddTensorOutputAction(const InputParameters & parameters);

  virtual void act() override;
};
