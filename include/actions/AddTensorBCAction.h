/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseObjectAction.h"

class TensorProblem;

/**
 * Adds a TensorBoundaryCondition object from a [TensorComputes][BCs] block.
 * Distinct from AddLBMBCAction, which handles the pre-existing, LBM-only
 * [TensorComputes][Boundary] block and is unaffected by this action.
 */
class AddTensorBCAction : public MooseObjectAction
{
public:
  static InputParameters validParams();

  AddTensorBCAction(const InputParameters & parameters);

  virtual void act() override;
};
