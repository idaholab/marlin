/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

class TensorPredictor;

/**
 * Interface for tensor solvers with internal iterations
 */
class IterativeTensorSolverInterface
{
public:
  IterativeTensorSolverInterface();

  const unsigned int & getIterations() const  { return _iterations; }
  const bool & isConverged() const { return _is_converged; }

  void addPredictor(std::shared_ptr<TensorPredictor> predictor);

protected:
  void applyPredictors();

  unsigned int _iterations;
  bool _is_converged;

  std::vector<std::shared_ptr<TensorPredictor>> _predictors;
};
