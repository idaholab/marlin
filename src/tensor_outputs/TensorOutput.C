/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "Moose.h"
#include "TensorOutput.h"
#include "MooseError.h"
#include "MarlinTypes.h"
#include "TensorProblem.h"
#include "DomainAction.h"

InputParameters
TensorOutput::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.addRequiredParam<std::vector<TensorInputBufferName>>("buffer", "The buffers to output");
  params.addParam<std::string>(
      "file_base",
      "The desired solution output name without an extension. If not provided, MOOSE sets it "
      "with Outputs/file_base when available. Otherwise, MOOSE uses input file name and this "
      "object name for a master input or uses master file_base, the subapp name and this object "
      "name for a subapp input to set it.");
  params.registerBase("TensorOutput");
  params.addPrivateParam<TensorProblem *>("_tensor_problem", nullptr);
  params.addPrivateParam<const DomainAction *>("_domain", nullptr);

  // Add the 'execute_on' input parameter for users to set
  ExecFlagEnum exec_enum;
  exec_enum.addAvailableFlags(EXEC_INITIAL);
  exec_enum.addAvailableFlags(EXEC_TIMESTEP_END);
  exec_enum = {EXEC_INITIAL, EXEC_TIMESTEP_END};
  params.addParam<ExecFlagEnum>("execute_on", exec_enum, exec_enum.getDocString());

  params.addClassDescription("TensorOutput object.");
  return params;
}

TensorOutput::TensorOutput(const InputParameters & parameters)
  : MooseObject(parameters),
    _tensor_problem(*getCheckedPointerParam<TensorProblem *>("_tensor_problem")),
    _domain(*getCheckedPointerParam<const DomainAction *>("_domain")),
    /* Outputs run in a dedicated thread. We must be careful not to access data from the problem
       class that might be updated while the output is running, which would lead to race conditions
       resulting in unreproducible outputs. Time is such a quantity, which is why we provide a
       dedicated output time that is not changed while the asynchronous output is running.*/
    _time(_tensor_problem.outputTime()),
    _file_base(isParamValid("file_base") ? getParam<std::string>("file_base")
                                         : _app.getOutputFileBase(true)),
    _execute_on(getParam<ExecFlagEnum>("execute_on"))
{
  for (const auto & name : getParam<std::vector<TensorInputBufferName>>("buffer"))
    _out_buffers[name] = &_tensor_problem.getBufferBase(name).getRawCPUTensor();
}

bool
TensorOutput::shouldRun(const ExecFlagType & execute_flag) const
{
  return _execute_on.isValueSet(execute_flag);
}

void
TensorOutput::startOutput()
{
  // allow derived classes to snapshot lightweight metadata before async output
  prepareForOutput();
  if (_output_thread.joinable())
    mooseError("Output thread is already running. Must call waitForCompletion() first. This is a "
               "code error.");
  _output_thread = std::move(std::thread(&TensorOutput::output, this));
}

void
TensorOutput::waitForCompletion()
{
  if (_output_thread.joinable())
    _output_thread.join();
}
