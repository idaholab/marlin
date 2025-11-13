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
 * This class adds an TensorOperator object.
 * The TensorOperator performs a mathematical operation on input tensors to produce output tensors
 */
class AddTensorComputeAction : public MooseObjectAction
{
public:
  static InputParameters validParams();

  AddTensorComputeAction(const InputParameters & parameters);

  virtual void act() override;
};
