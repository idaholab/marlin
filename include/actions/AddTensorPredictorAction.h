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
 * This class adds an TensorPredictor object to the current solver.
 */
class AddTensorPredictorAction : public MooseObjectAction
{
public:
  static InputParameters validParams();

  AddTensorPredictorAction(const InputParameters & parameters);

  virtual void act() override;
};
