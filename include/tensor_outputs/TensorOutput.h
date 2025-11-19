/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseObject.h"
#include <thread>
#include <torch/torch.h>

class TensorBufferBase;
class TensorProblem;
class DomainAction;

/**
 * Direct buffer output
 */
class TensorOutput : public MooseObject
{
public:
  static InputParameters validParams();

  TensorOutput(const InputParameters & parameters);

  virtual void init() {}

  virtual bool shouldRun(const ExecFlagType & execute_flag) const;

  void startOutput();
  void waitForCompletion();

protected:
  virtual void output() = 0;

  TensorProblem & _tensor_problem;
  const DomainAction & _domain;

  /// simulation time of the step that is being output
  const Real & _time;

  /// output file name prefix
  const std::string _file_base;

  std::thread _output_thread;

  /// The buffer this output object is outputting
  std::map<std::string, const torch::Tensor *> _out_buffers;

  const ExecFlagEnum _execute_on;
};
