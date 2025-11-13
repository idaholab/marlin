/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorSolver.h"
#include "TensorPredictor.h"
#include "TensorProblem.h"

IterativeTensorSolverInterface::IterativeTensorSolverInterface()
  :  _iterations(0),
    _is_converged(true)
{
}

void
IterativeTensorSolverInterface::applyPredictors()
{
  for (const auto & pred : _predictors)
    pred->computeBuffer();
}
