/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "FEProblem.h"
#include "DomainInterface.h"
#include "InputParameters.h"
#include "MarlinTypes.h"
#include "MarlinUtils.h"
#include "TensorBuffer.h"

// list tensor buffer includes here
#include "PlainTensorBuffer.h"
#include "NEML2TensorBuffer.h"

#include "AuxiliarySystem.h"
#include "libmesh/petsc_vector.h"
#include "libmesh/print_trace.h"

#include <cstddef>
#include <list>
#include <memory>
#include <torch/torch.h>
#include <type_traits>

class UniformTensorMesh;
class TensorOperatorBase;
template <typename T = torch::Tensor>
class TensorTimeIntegrator;
class TensorOutput;
class TensorSolver;
class CreateTensorSolverAction;

namespace Marlin
{
struct ConstantBase
{
  virtual ~ConstantBase() = default;
  virtual std::string getType() = 0;
};

template <typename T>
struct Constant : public ConstantBase
{
  virtual std::string getType() override { return libMesh::demangle(typeid(T).name()); }
  T _value;
};
}

/**
 * Problem for solving eigenvalue problems
 */
class TensorProblem : public FEProblem, public DomainInterface
{
public:
  static InputParameters validParams();

  TensorProblem(const InputParameters & parameters);
  ~TensorProblem() override;

  // setup stuff
  void init() override;

  // run compute objects
  void execute(const ExecFlagType & exec_type) override;

  // move tensors in time
  void advanceState() override;

  // recompute quantities on grid size change
  virtual void gridChanged();

  virtual void addTensorBuffer(const std::string & buffer_type,
                               const std::string & buffer_name,
                               InputParameters & parameters);

  template <typename T>
  std::shared_ptr<TensorBuffer<T>> addTensorBuffer(const std::string & buffer_name);

  virtual void addTensorComputeInitialize(const std::string & compute_name,
                                          const std::string & name,
                                          InputParameters & parameters);
  virtual void addTensorComputeSolve(const std::string & compute_name,
                                     const std::string & name,
                                     InputParameters & parameters);
  virtual void addTensorComputePostprocess(const std::string & compute_name,
                                           const std::string & name,
                                           InputParameters & parameters);

  virtual void addTensorOutput(const std::string & output_name,
                               const std::string & name,
                               InputParameters & parameters);

  /// check if a tensor buffer with the given type exists
  template <typename T>
  bool hasBuffer(const std::string & buffer_name);

  /// returns the current state of the tensor
  template <typename T = torch::Tensor>
  T & getBuffer(const std::string & buffer_name);

  /// requests a tensor regardless of type
  TensorBufferBase & getBufferBase(const std::string & buffer_name);

  /// return the old states of the tensor
  template <typename T = torch::Tensor>
  const std::vector<T> & getBufferOld(const std::string & buffer_name, unsigned int max_states);

  /// returns a reference to a raw torch::Tensor view of buffer_name
  const torch::Tensor & getRawBuffer(const std::string & buffer_name);

  /// returns a reference to a copy of buffer_name that is guaranteed to be contiguous and located on the CPU device
  const torch::Tensor & getRawCPUBuffer(const std::string & buffer_name);

  virtual Real & subDt() { return _sub_dt; }
  virtual Real & subTime() { return _sub_time; }
  virtual Real & outputTime() { return _output_time; }

  /// align a 1d tensor in a specific dimension
  torch::Tensor align(torch::Tensor t, unsigned int dim) const;

  /// get the domain shape (to build tensors from scratch) TODO: make sure this is local
  const torch::IntArrayRef & getShape() { return _shape; }

  typedef std::vector<std::shared_ptr<TensorOperatorBase>> TensorComputeList;
  const TensorComputeList & getComputes() const { return _computes; }

  template <typename T = TensorOperatorBase>
  T & getCompute(const std::string & name) const;

  typedef std::vector<std::shared_ptr<TensorOutput>> TensorOutputList;
  const TensorOutputList & getOutputs() const { return _outputs; }

  template <typename T>
  const T & getConstant(const std::string & name);

  template <typename T>
  void declareConstant(const std::string & name, const T & value);

  /// The CreateTensorSolverAction calls this to set the active solver
  void setSolver(std::shared_ptr<TensorSolver> solver,
                 const MooseTensor::Key<CreateTensorSolverAction> &);

  /// get a reference to the current solver
  template <typename T>
  T & getSolver() const;

  static TensorProblem & cast(MooseObject * moose_object, Problem & problem);

protected:
  void updateDOFMap();

  template <typename FLOAT_TYPE>
  void mapBuffersToAux();

  template <typename FLOAT_TYPE>
  void mapAuxToBuffers();

  virtual void addTensorCompute(const std::string & compute_name,
                                const std::string & name,
                                InputParameters & parameters,
                                TensorComputeList & list);

  /// execute initial conditionobjects
  void executeTensorInitialConditions();

  /// perform output tasks
  void executeTensorOutputs(const ExecFlagType & exec_type);

  /// helper to get the TensorBuffer wrapper object that holds the actual tensor data
  template <typename T = torch::Tensor>
  TensorBuffer<T> & getBufferHelper(const std::string & buffer_name);

  /// tensor options
  const torch::TensorOptions _options;

  /// show debug ouput
  const bool _debug;

  /// solver substeps
  const unsigned int _substeps;

  /// substepping timestep
  Real _sub_dt;
  Real _sub_time;

  /// simulation time for the currently running output thread
  Real _output_time;

  /// list of TensorBuffers (i.e. tensors)
  std::map<std::string, std::shared_ptr<TensorBufferBase>> _tensor_buffer;

  /// old timesteps
  std::vector<Real> _old_dt;

  const unsigned int & _dim;

  /// grid spacing
  const RealVectorValue & _grid_spacing;

  /// global grid size
  const std::array<int64_t, 3> & _n;

  /// domain shape
  const torch::IntArrayRef & _shape;

  /// solve objects
  TensorComputeList _computes;

  /// initialization objects
  TensorComputeList _ics;

  /// postprocessing objects
  TensorComputeList _pps;

  /// on demand objects that are explicitly triggered by other objects
  TensorComputeList _on_demand;

  ///  time integrator objects
  std::vector<std::shared_ptr<TensorTimeIntegrator<>>> _time_integrators;

  std::vector<std::shared_ptr<TensorOutput>> _outputs;

  /// map from buffer name to variable name
  std::map<std::string, AuxVariableName> _buffer_to_var_name;

  /// buffers to solution vector indices
  std::map<std::string, std::tuple<const MooseVariableFieldBase *, std::vector<std::size_t>, bool>>
      _buffer_to_var;
  std::map<std::string, std::tuple<const MooseVariableFieldBase *, std::vector<std::size_t>, bool>>
      _var_to_buffer;

  /// The [TensorSolver]
  std::shared_ptr<TensorSolver> _solver;

  /// parameters
  std::map<std::string, std::unique_ptr<Marlin::ConstantBase>> _constants;
  std::set<std::string> _fetched_constants;
  std::list<Real> _literal_constants;
  bool _can_fetch_constants;

private:
  std::list<InputParameters> _buffer_params;
};

template <typename T>
T &
TensorProblem::getSolver() const
{
  if (_solver)
  {
    const auto specialized_solver = dynamic_cast<T *>(_solver.get());
    if (specialized_solver)
      return *specialized_solver;
    mooseError(
        "No TensorSolver supporting the requested type '", typeid(T).name(), "' has been set up.");
  }
  mooseError("No TensorSolver has been set up.");
}

template <typename T>
bool
TensorProblem::hasBuffer(const std::string & buffer_name)
{
  auto it = _tensor_buffer.find(buffer_name);
  return (it != _tensor_buffer.end() && !dynamic_cast<TensorBuffer<T> *>(it->second.get()));
}

template <typename T>
TensorBuffer<T> &
TensorProblem::getBufferHelper(const std::string & buffer_name)
{
  auto it = _tensor_buffer.find(buffer_name);

  if (it == _tensor_buffer.end())
    return *addTensorBuffer<T>(buffer_name);
  else
  {
    auto tensor_buffer = dynamic_cast<TensorBuffer<T> *>(it->second.get());
    if (!tensor_buffer)
      mooseError("TensorBuffer '",
                 buffer_name,
                 "' of the requested type '",
                 libMesh::demangle(typeid(T).name()),
                 "' was previously declared as '",
                 it->second->type(),
                 "'.");
    return *tensor_buffer;
  }
}

template <typename T>
T &
TensorProblem::getBuffer(const std::string & buffer_name)
{
  return getBufferHelper<T>(buffer_name).getTensor();
}

template <typename T>
const std::vector<T> &
TensorProblem::getBufferOld(const std::string & buffer_name, unsigned int max_states)
{
  return getBufferHelper<T>(buffer_name).getOldTensor(max_states);
}

template <typename T>
std::shared_ptr<TensorBuffer<T>>
TensorProblem::addTensorBuffer(const std::string & buffer_name)
{
  if (_debug)
    mooseInfoRepeated("Automatically adding tensor '",
                      buffer_name,
                      "' of type '",
                      libMesh::demangle(typeid(T).name()),
                      "'");

  _buffer_params.push_back(TensorBufferSpecialization<T>::type::validParams());
  auto & params = _buffer_params.back();
  params.template set<std::string>("_object_name") = buffer_name;
  params.template set<FEProblem *>("_fe_problem") = this;
  params.template set<FEProblemBase *>("_fe_problem_base") = this;
  params.template set<THREAD_ID>("_tid") = 0;
  params.template set<std::string>("_type") = "TensorBufferBase";
  params.template set<MooseApp *>("_moose_app") = &getMooseApp();
  params.finalize(buffer_name);
  auto tensor_buffer = std::make_shared<typename TensorBufferSpecialization<T>::type>(params);

  _tensor_buffer.try_emplace(buffer_name, tensor_buffer);
  return tensor_buffer;
}

template <typename T>
const T &
TensorProblem::getConstant(const std::string & name)
{
  if (!_can_fetch_constants)
    mooseError("Constants must be gotten during object construction.");

  // deal with literal scalars
  if constexpr (std::is_same_v<T, Real>)
  {
    // check if we are able to convert the name into a Real
    Real real_value;
    std::istringstream ss(name);
    if (ss >> real_value && ss.eof())
    {
      _literal_constants.push_back(real_value);
      return _literal_constants.back();
    }
  }

  const auto it = _constants.find(name);
  if (it != _constants.end())
  {
    auto * t_param = dynamic_cast<Marlin::Constant<T> *>(it->second.get());
    if (t_param == nullptr)
      mooseError("Fetching constant '",
                 name,
                 "' (which is of type ",
                 it->second->getType(),
                 ") with the wrong type (",
                 libMesh::demangle(typeid(T).name()),
                 ").");
    return t_param->_value;
  }
  else
  {
    auto param = std::make_unique<Marlin::Constant<T>>();
    const auto & ref = param->_value;
    _constants[name] = std::move(param);
    _fetched_constants.insert(name);
    return ref;
  }
}

template <typename T>
void
TensorProblem::declareConstant(const std::string & name, const T & value)
{
  if (!_can_fetch_constants)
    mooseError("Constants must be declared during object construction.");

  const auto it = _constants.find(name);
  if (it != _constants.end())
  {
    auto * t_param = dynamic_cast<Marlin::Constant<T> *>(it->second.get());
    if (t_param == nullptr)
      mooseError("Declaring constant '",
                 name,
                 "' (which was already fetched as type ",
                 it->second->getType(),
                 ") with the wrong type (",
                 libMesh::demangle(typeid(T).name()),
                 ").");

    if (_fetched_constants.count(name))
    {
      // placeholder parameter created by getParameter
      t_param->_value = value;
      _fetched_constants.erase(name);
    }
    else
      mooseError("Parameter '", name, "' has already been declared.");
  }
  else
  {
    auto param = std::make_unique<Marlin::Constant<T>>();
    param->_value = value;
    _constants[name] = std::move(param);
  }
}

template <typename T>
T &
TensorProblem::getCompute(const std::string & name) const
{
  for (const auto & tcb : _computes)
    if (std::dynamic_pointer_cast<MooseObject>(tcb)->name() == name)
      if (const auto ptr = std::dynamic_pointer_cast<T>(tcb); ptr)
        return *ptr;

  mooseError("No compute with name '",
             name,
             "' of type ",
             libMesh::demangle(typeid(T).name()),
             " found.");
}
