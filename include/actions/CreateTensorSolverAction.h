/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseObjectAction.h"
#include "DomainInterface.h"

/**
 * This class creates the TensorSolver object.
 */
class CreateTensorSolverAction : public MooseObjectAction, public DomainInterface
{
public:
  static InputParameters validParams();

  CreateTensorSolverAction(const InputParameters & parameters);

  virtual void act() override;
};
