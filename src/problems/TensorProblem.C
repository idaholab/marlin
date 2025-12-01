/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "Moose.h"
#include "MooseError.h"
#include "TensorProblem.h"
#include "TensorSolver.h"
#include "UniformTensorMesh.h"

#include "TensorOperatorBase.h"
#include "TensorTimeIntegrator.h"
#include "TensorOutput.h"
#include "DomainAction.h"

#include "MarlinUtils.h"
#include "DependencyResolverInterface.h"
#include <memory>

registerMooseObject("MarlinApp", TensorProblem);

InputParameters
TensorProblem::validParams()
{
  InputParameters params = FEProblem::validParams();
  params.addClassDescription(
      "A normal Problem object that adds the ability to perform spectral solves.");
  params.set<bool>("skip_nl_system_check") = true;
  params.addParam<bool>("print_debug_output", false, "Show Tensor specific debug outputs");
  params.addParam<unsigned int>(
      "spectral_solve_substeps",
      1,
      "How many substeps to divide the spectral solve for each MOOSE timestep into.");
  params.addParam<std::vector<std::string>>("scalar_constant_names", "Scalar constant names");
  params.addParam<std::vector<Real>>("scalar_constant_values", "Scalar constant values");
  return params;
}

TensorProblem::TensorProblem(const InputParameters & parameters)
  : FEProblem(parameters),
    DomainInterface(this),
    _options(MooseTensor::floatTensorOptions()),
    _debug(getParam<bool>("print_debug_output")),
    _substeps(getParam<unsigned int>("spectral_solve_substeps")),
    _dim(_domain.getDim()),
    _grid_spacing(_domain.getGridSpacing()),
    _n((_domain.getGridSize())),
    _shape(_domain.getShape()),
    _solver(nullptr),
    _can_fetch_constants(true),
    _local_tensor_shape(_domain.getLocalGridSize())
{
  // get constants (for scalar constants we provide a shortcut in the problem block)
  for (const auto & [name, value] :
       getParam<std::string, Real>("scalar_constant_names", "scalar_constant_values"))
    declareConstant<Real>(name, value);

  // make sure AuxVariables are contiguous in the solution vector
  getAuxiliarySystem().sys().identify_variable_groups(false);
}

TensorProblem::~TensorProblem()
{
  // wait for outputs to be completed (otherwise resources might get freed that the output thread
  // depends on)
  for (auto & output : _outputs)
    output->waitForCompletion();
}

void
TensorProblem::init()
{
  unsigned int n_threads = libMesh::n_threads();
  if (n_threads != 1)
  {
    mooseInfo("Setting libTorch to use ", n_threads, " threads on the CPU.");
    torch::set_num_threads(n_threads);
  }

  updateLocalTensorShape();

  // initialize tensors
  for (auto pair : _tensor_buffer)
    pair.second->init();

  // compute grid dependent quantities
  gridChanged();

  // init computes (must happen before dependency update)
  for (auto & initializer : _ics)
    initializer->init();

  // init computes (must happen before dependency update)
  for (auto & cmp : _computes)
    cmp->init();

  // update dependencies
  if (_solver)
    _solver->updateDependencies();
  else
  {
    // dependency resolution of TensorComputes
    DependencyResolverInterface::sort(_computes);
  }

  // dependency resolution of TensorICs
  DependencyResolverInterface::sort(_ics);

  // dependency resolution of Tensor Postprocessors
  DependencyResolverInterface::sort(_pps);

  // show computes
  if (_debug)
  {
    _console << COLOR_CYAN << "Compute object execution order:\n" << COLOR_DEFAULT;
    for (auto & cmp : _computes)
    {
      _console << "  " << cmp->name() << '\n' << COLOR_YELLOW;
      for (const auto & ri : cmp->getRequestedItems())
        _console << "    <- " << ri << '\n';
      _console << COLOR_GREEN;
      for (const auto & si : cmp->getSuppliedItems())
        _console << "    -> " << si << '\n';
      _console << COLOR_DEFAULT;
    }
  }

  // call base class init
  FEProblem::init();

  // init outputs
  for (auto & output : _outputs)
    output->init();

  // check computes (after dependency update)
  for (auto & cmp : _computes)
    cmp->check();

  updateDOFMap();

  // debug output
  std::string variable_mapping;
  for (const auto & [buffer_name, tuple] : _buffer_to_var)
    variable_mapping += (std::get<bool>(tuple) ? "NODAL     " : "ELEMENTAL ") + buffer_name + '\n';
  if (!variable_mapping.empty())
    mooseInfo("Direct buffer to solution vector mappings:\n", variable_mapping);
}

void
TensorProblem::execute(const ExecFlagType & exec_type)
{
  if (exec_type == EXEC_INITIAL)
  {
    // check for constants
    if (_fetched_constants.size() == 1)
      mooseError(
          "Constant ", Moose::stringify(_fetched_constants), " was requested but never declared.");
    if (_fetched_constants.size() > 1)
      mooseError("Constants ",
                 Moose::stringify(_fetched_constants),
                 " were requested but never declared.");
    _can_fetch_constants = false;

    // update time
    _sub_time = FEProblem::time();

    executeTensorInitialConditions();

    executeTensorOutputs(EXEC_INITIAL);
  }

  if (exec_type == EXEC_TIMESTEP_BEGIN)
  {
    // update time
    _sub_time = FEProblem::timeOld();

    // run solver
    if (_solver)
      runComputeWithGhosts(*_solver);
    else
      for (auto & cmp : _computes)
        runComputeWithGhosts(*cmp);
  }

  if (exec_type == EXEC_TIMESTEP_END)
  {
    // run outputs
    executeTensorOutputs(EXEC_TIMESTEP_END);
  }

  FEProblem::execute(exec_type);
}

void
TensorProblem::executeTensorInitialConditions()
{
  // run ICs
  for (auto & ic : _ics)
    runComputeWithGhosts(*ic);

  // compile ist of compute output tensors
  std::set<std::string> _is_output;
  for (auto & cmp : _computes)
    _is_output.insert(cmp->getSuppliedItems().begin(), cmp->getSuppliedItems().end());

  // // check for uninitialized tensors
  // for (auto & [name, t] : _tensor_buffer)
  //   if (!t.defined() && _is_output.count(name) == 0)
  //     mooseWarning(name, " is not initialized and not an output of any [Solve] compute.");
}

/// perform output tasks
void
TensorProblem::executeTensorOutputs(const ExecFlagType & exec_flag)
{
  // run postprocessing before output
  for (auto & pp : _pps)
    runComputeWithGhosts(*pp);

  // wait for prior asynchronous activity on CPU buffers to complete
  // (this is a synchronization barrier for the threaded CPU activity)
  for (auto & output : _outputs)
    output->waitForCompletion();

  // update output time
  _output_time = _time;

  // prepare CPU buffers (this is a synchronization barrier for the GPU)
  for (const auto & pair : _tensor_buffer)
    pair.second->makeCPUCopy();

  // run direct buffer outputs (asynchronous in threads)
  for (auto & output : _outputs)
    if (output->shouldRun(exec_flag))
      output->startOutput();

  if (_options.dtype() == torch::kFloat64)
    mapBuffersToAux<double>();
  else if (_options.dtype() == torch::kFloat32)
    mapBuffersToAux<float>();
  else
    mooseError("torch::Dtype unsupported by mapBuffersToAux.");
}

void
TensorProblem::updateDOFMap()
{
  TIME_SECTION("update", 3, "Updating Tensor DOF Map", true);
  const auto & min_global = _domain.getDomainMin();

  // variable mapping
  const auto & aux = getAuxiliarySystem();
  if (!const_cast<libMesh::System &>(aux.system()).is_initialized())
    mooseError("Aux system is not initialized :(");

  auto sys_num = aux.number();
  for (auto & [buffer_name, tuple] : _buffer_to_var)
  {
    auto & [var, dofs, is_nodal] = tuple;
    if (var->isArray() || var->isVector() || var->isFV())
      mooseError("Unsupported variable type for mapping");
    auto var_num = var->number();

    auto compute_iteration_index = [this](Point p, long int n0, long int n1)
    {
      return static_cast<long int>(p(0) / _grid_spacing(0)) +
             (_dim > 1 ? static_cast<long int>(p(1) / _grid_spacing(1)) * n0 : 0) +
             (_dim > 2 ? static_cast<long int>(p(2) / _grid_spacing(2)) * n0 * n1 : 0);
    };

    if (is_nodal)
    {
      long int n0 = _n[0] + 1;
      long int n1 = _n[1] + 1;
      long int n2 = _n[2] + 1;
      dofs.resize(n0 * (_dim > 1 ? n1 : 1) * (_dim > 2 ? n2 : 1));

      // loop over nodes
      const static Point shift = _grid_spacing / 2.0 - min_global;
      for (const auto & node : _mesh.getMesh().node_ptr_range())
      {
        const auto dof_index = node->dof_number(sys_num, var_num, 0);
        const auto iteration_index = compute_iteration_index(*node + shift, n0, n1);
        dofs[iteration_index] = dof_index;
      }
    }
    else
    {
      long int n0 = _n[0];
      long int n1 = _n[1];
      long int n2 = _n[2];
      dofs.resize(n0 * n1 * n2);

      // loop over elements
      const static Point shift = -min_global;
      for (const auto & elem : _mesh.getMesh().element_ptr_range())
      {
        const auto dof_index = elem->dof_number(sys_num, var_num, 0);
        const auto iteration_index =
            compute_iteration_index(elem->vertex_average() + shift, n0, n1);
        dofs[iteration_index] = dof_index;
      }
    }
  }
}

template <typename FLOAT_TYPE>
void
TensorProblem::mapBuffersToAux()
{
  // nothing to map?
  if (_buffer_to_var.empty())
    return;

  TIME_SECTION("update", 3, "Mapping Tensor buffers to Variables", true);

  auto * current_solution = &getAuxiliarySystem().solution();
  auto * solution_vector = dynamic_cast<PetscVector<Number> *>(current_solution);
  if (!solution_vector)
    mooseError(
        "Cannot map directly to the solution vector because NumericVector is not a PetscVector!");

  auto value = solution_vector->get_array();

  // const monomial variables
  for (const auto & [buffer_name, tuple] : _buffer_to_var)
  {
    const auto & [var, dofs, is_nodal] = tuple;
    libmesh_ignore(var);
    const long int n0 = is_nodal ? _n[0] + 1 : _n[0];
    const long int n1 = is_nodal ? _n[1] + 1 : _n[1];
    const long int n2 = is_nodal ? _n[2] + 1 : _n[2];

    // TODO: better design that works for NEML2 tensors as well
    const auto buffer = getRawCPUBuffer(buffer_name);
    if (buffer.sizes().size() != _dim)
      mooseError("Buffer '",
                 buffer_name,
                 "' is not a scalar tensor field and is not yet supported for AuxVariable mapping");
    std::size_t idx = 0;
    switch (_dim)
    {
      {
        case 1:
          const auto b = buffer.template accessor<FLOAT_TYPE, 1>();
          for (const auto i : make_range(n0))
            value[dofs[idx++]] = b[i % _n[0]];
          break;
      }
      case 2:
      {
        const auto b = buffer.template accessor<FLOAT_TYPE, 2>();
        for (const auto j : make_range(n1))
          for (const auto i : make_range(n0))
            value[dofs[idx++]] = b[i % _n[0]][j % _n[1]];
        break;
      }
      case 3:
      {
        const auto b = buffer.template accessor<FLOAT_TYPE, 3>();
        for (const auto k : make_range(n2))
          for (const auto j : make_range(n1))
            for (const auto i : make_range(n0))
              value[dofs[idx++]] = b[i % _n[0]][j % _n[1]][k % _n[2]];
        break;
      }
      default:
        mooseError("Unsupported dimension");
    }
  }

  solution_vector->restore_array();
  getAuxiliarySystem().sys().update();
}

template <typename FLOAT_TYPE>
void
TensorProblem::mapAuxToBuffers()
{
  // nothing to map?
  if (_var_to_buffer.empty())
    return;

  TIME_SECTION("update", 3, "Mapping Variables to Tensor buffers", true);

  const auto * current_solution = &getAuxiliarySystem().solution();
  const auto * solution_vector = dynamic_cast<const PetscVector<Number> *>(current_solution);
  if (!solution_vector)
    mooseError(
        "Cannot map directly to the solution vector because NumericVector is not a PetscVector!");

  const auto value = solution_vector->get_array_read();

  // const monomial variables
  for (const auto & [buffer_name, tuple] : _var_to_buffer)
  {
    const auto & [var, dofs, is_nodal] = tuple;
    libmesh_ignore(var);
    const auto buffer = getBufferBase(buffer_name).getRawCPUTensor();
    std::size_t idx = 0;
    switch (_dim)
    {
      {
        case 1:
          auto b = buffer.template accessor<FLOAT_TYPE, 1>();
          for (const auto i : make_range(_n[0]))
            b[i % _n[0]] = value[dofs[idx++]];
          break;
      }
      case 2:
      {
        auto b = buffer.template accessor<FLOAT_TYPE, 2>();
        for (const auto j : make_range(_n[1]))
        {
          for (const auto i : make_range(_n[0]))
            b[i % _n[0]][j % _n[1]] = value[dofs[idx++]];
          if (is_nodal)
            idx++;
        }
        break;
      }
      case 3:
      {
        auto b = buffer.template accessor<FLOAT_TYPE, 3>();
        for (const auto k : make_range(_n[2]))
        {
          for (const auto j : make_range(_n[1]))
          {
            for (const auto i : make_range(_n[0]))
              b[i % _n[0]][j % _n[1]][k % _n[2]] = value[dofs[idx++]];
            if (is_nodal)
              idx++;
          }
          if (is_nodal)
            idx += _n[0] + 1;
        }
        break;
      }
      default:
        mooseError("Unsupported dimension");
    }
  }
}

void
TensorProblem::advanceState()
{
  FEProblem::advanceState();

  if (timeStep() <= 1)
    return;

  // move buffers in time
  std::size_t total_max = 0;
  for (auto & pair : _tensor_buffer)
    total_max = std::max(total_max, pair.second->advanceState());

  // move dt in time (UGH, we need the _substep_dt!!!!)
  if (_old_dt.size() < total_max)
    _old_dt.push_back(0.0);
  if (!_old_dt.empty())
  {
    for (std::size_t i = _old_dt.size() - 1; i > 0; --i)
      _old_dt[i] = _old_dt[i - 1];
    _old_dt[0] = _dt;
  }
}

void
TensorProblem::updateLocalTensorShape()
{
  const auto & owned = _domain.getLocalGridSize();
  for (const auto d : {0u, 1u, 2u})
    _local_tensor_shape[d] = owned[d] + 2 * static_cast<int64_t>(_max_ghost_layers);
}

void
TensorProblem::gridChanged()
{
  updateLocalTensorShape();
  // _domain.gridChanged();
}

void
TensorProblem::addTensorBuffer(const std::string & buffer_type,
                               const std::string & buffer_name,
                               InputParameters & parameters)
{
  if (_domain.isRealSpaceMode() && parameters.get<bool>("reciprocal"))
    mooseError("Reciprocal space tensors are not supported in REAL_SPACE parallel mode.");

  // add buffer
  if (_tensor_buffer.find(buffer_name) != _tensor_buffer.end())
    mooseError("TensorBuffer '", buffer_name, "' already exists in the system");

  // Add a pointer to the TensorProblem and the Domain
  parameters.addPrivateParam<TensorProblem *>("_tensor_problem", this);
  // parameters.addPrivateParam<const DomainAction *>("_domain", &_domain);

  // Create the object
  auto tensor_buffer = _factory.create<TensorBufferBase>(buffer_type, buffer_name, parameters, 0);
  logAdd("TensorBufferBase", buffer_name, buffer_type, parameters);

  _tensor_buffer.try_emplace(buffer_name, tensor_buffer);

  // store variable mapping
  const auto & var_names = parameters.get<std::vector<AuxVariableName>>("map_to_aux_variable");
  if (!var_names.empty())
  {
    const auto & aux = getAuxiliarySystem();
    const auto var_name = var_names[0];
    if (!aux.hasVariable(var_name))
      mooseError("AuxVariable '", var_name, "' does not exist in the system.");

    bool is_nodal;
    const auto & var = aux.getVariable(0, var_name);
    if (var.feType() == FEType(FIRST, LAGRANGE))
      is_nodal = true;
    else if (var.feType() == FEType(CONSTANT, MONOMIAL))
      is_nodal = false;
    else
      mooseError("Only first order lagrange and constant monomial variables are supported for "
                 "direct transfer. Try using the ProjectTensorAux kernel to transfer buffers to "
                 "variables of any other type.");

    _buffer_to_var[buffer_name] = std::make_tuple(&var, std::vector<std::size_t>{}, is_nodal);

    // call this to mark the CPU copy as requested
    getRawCPUBuffer(buffer_name);
  }
}

void
TensorProblem::addTensorComputeSolve(const std::string & compute_type,
                                     const std::string & compute_name,
                                     InputParameters & parameters)
{
  addTensorCompute(compute_type, compute_name, parameters, _computes);
}

void
TensorProblem::addTensorComputeInitialize(const std::string & compute_type,
                                          const std::string & compute_name,
                                          InputParameters & parameters)
{
  addTensorCompute(compute_type, compute_name, parameters, _ics);
}

void
TensorProblem::addTensorComputePostprocess(const std::string & compute_name,
                                           const std::string & name,
                                           InputParameters & parameters)
{
  addTensorCompute(compute_name, name, parameters, _pps);
}

void
TensorProblem::addTensorCompute(const std::string & compute_type,
                                const std::string & compute_name,
                                InputParameters & parameters,
                                TensorComputeList & list)
{
  // Add a pointer to the TensorProblem and the Domain
  parameters.addPrivateParam<TensorProblem *>("_tensor_problem", this);
  parameters.addPrivateParam<const DomainAction *>("_domain", &_domain);

  // Create the object
  auto compute_object =
      _factory.create<TensorOperatorBase>(compute_type, compute_name, parameters, 0);
  logAdd("TensorOperatorBase", compute_name, compute_type, parameters);
  list.push_back(compute_object);
}

void
TensorProblem::addTensorOutput(const std::string & output_type,
                               const std::string & output_name,
                               InputParameters & parameters)
{
  // Add a pointer to the TensorProblem and the Domain
  parameters.addPrivateParam<TensorProblem *>("_tensor_problem", this);
  parameters.addPrivateParam<const DomainAction *>("_domain", &_domain);

  // Create the object
  auto output_object = _factory.create<TensorOutput>(output_type, output_name, parameters, 0);
  logAdd("TensorOutput", output_name, output_type, parameters);
  _outputs.push_back(output_object);
}

void
TensorProblem::exchangeGhostLayers(const std::string & buffer_name, unsigned int ghost_layers)
{
  if (ghost_layers == 0)
    return;

  if (!_domain.isRealSpaceMode())
    mooseError("Ghost exchange is only supported in REAL_SPACE parallel mode.");

  if (ghost_layers > _max_ghost_layers)
    mooseError("Requested ghost layers (",
               ghost_layers,
               ") exceed allocated halo width (",
               _max_ghost_layers,
               ").");

  auto & base = getBufferBase(buffer_name);
  auto * tensor_buffer = dynamic_cast<TensorBuffer<torch::Tensor> *>(&base);
  if (!tensor_buffer)
    mooseError("Ghost exchange supports torch::Tensor buffers only (buffer '", buffer_name, "').");

  auto & tensor = tensor_buffer->getTensor();
  const auto mpi_type = _domain.getMPIType(tensor.scalar_type());
  const auto partitions = _domain.getRealSpacePartitions();
  const auto index = _domain.getRealSpaceIndex();
  const auto periodic = _domain.getPeriodicDirections();
  const auto owned = _domain.getLocalGridSize();

  for (unsigned int d = 0; d < _dim; ++d)
    if (ghost_layers > static_cast<unsigned int>(owned[d]))
      mooseError("Requested ghost layers (",
                 ghost_layers,
                 ") exceed owned extent (",
                 owned[d],
                 ") in direction ",
                 d,
                 ".");

  const bool use_gpu = _domain.gpuAwareMPI() && tensor.is_cuda();
  const auto device_options = tensor.options();
  const auto cpu_options = tensor.options().device(torch::kCPU);

  const auto toRank = [&](unsigned int ix, unsigned int iy, unsigned int iz)
  { return static_cast<int>(ix + partitions[0] * (iy + partitions[1] * iz)); };

  const int64_t halo = static_cast<int64_t>(_max_ghost_layers);

  for (unsigned int d = 0; d < _dim; ++d)
  {
    const unsigned int part = partitions[d];
    if (part == 1)
    {
      if (periodic[d] && halo >= static_cast<int64_t>(ghost_layers))
      {
        const auto owned_width = owned[d];
        auto wrap_upper = tensor.narrow(d, halo, ghost_layers);
        auto wrap_lower = tensor.narrow(d, halo + owned_width - ghost_layers, ghost_layers);
        tensor.narrow(d, halo - ghost_layers, ghost_layers).copy_(wrap_lower);
        tensor.narrow(d, halo + owned_width, ghost_layers).copy_(wrap_upper);
      }
      continue;
    }

    const bool has_lower = index[d] > 0;
    const bool has_upper = (index[d] + 1) < part;

    struct NeighborExchange
    {
      int neighbor_rank;
      bool lower;
      torch::Tensor send_buf;
      torch::Tensor recv_buf;
      torch::Tensor recv_view;
      int send_tag;
      int recv_tag;
    };

    std::vector<NeighborExchange> exchanges;
    exchanges.reserve(2);

    auto prepare_neighbor = [&](bool lower)
    {
      std::array<unsigned int, 3> neighbor = index;
      if (lower)
      {
        if (has_lower)
          neighbor[d] = index[d] - 1;
        else if (periodic[d])
          neighbor[d] = part - 1;
        else
          return;
      }
      else
      {
        if (has_upper)
          neighbor[d] = index[d] + 1;
        else if (periodic[d])
          neighbor[d] = 0;
        else
          return;
      }

      const int neighbor_rank = toRank(neighbor[0], neighbor[1], neighbor[2]);
      const int64_t send_start =
          lower ? halo : halo + owned[d] - static_cast<int64_t>(ghost_layers);
      const int64_t recv_start =
          lower ? halo - static_cast<int64_t>(ghost_layers) : halo + owned[d];

      auto send_slice = tensor.narrow(d, send_start, ghost_layers).contiguous();
      auto recv_view = tensor.narrow(d, recv_start, ghost_layers);

      if (neighbor_rank == static_cast<int>(_domain.comm().rank()))
      {
        // self-wrap: copy directly, no MPI
        recv_view.copy_(send_slice);
        return;
      }

      auto send_buf = use_gpu ? send_slice : send_slice.to(cpu_options);
      auto recv_buf = torch::empty_like(send_buf, use_gpu ? device_options : cpu_options);

      const int send_tag = 200 + d * 2 + (lower ? 0 : 1);
      const int recv_tag = 200 + d * 2 + (lower ? 1 : 0);

      exchanges.push_back(
          {neighbor_rank, lower, send_buf, recv_buf, recv_view, send_tag, recv_tag});

      if (_domain.debug())
        mooseInfoRepeated("Ghost exchange d=",
                          d,
                          " lower=",
                          lower,
                          " me=",
                          _domain.comm().rank(),
                          " <-> ",
                          neighbor_rank,
                          " send_start=",
                          send_start,
                          " recv_start=",
                          recv_start,
                          " count=",
                          send_buf.numel());
    };

    prepare_neighbor(true);
    prepare_neighbor(false);

    if (!exchanges.empty())
    {
      std::vector<MPI_Request> reqs(2 * exchanges.size(), MPI_REQUEST_NULL);
      std::size_t idx = 0;
      // post all receives
      for (const auto & ex : exchanges)
        MPI_Irecv(ex.recv_buf.data_ptr(),
                  ex.recv_buf.numel(),
                  mpi_type,
                  ex.neighbor_rank,
                  ex.recv_tag,
                  _domain.getMPIComm(),
                  &reqs[idx++]);
      // post all sends
      for (const auto & ex : exchanges)
        MPI_Isend(ex.send_buf.data_ptr(),
                  ex.send_buf.numel(),
                  mpi_type,
                  ex.neighbor_rank,
                  ex.send_tag,
                  _domain.getMPIComm(),
                  &reqs[idx++]);

      MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

      for (auto & ex : exchanges)
      {
        if (!use_gpu)
          ex.recv_buf = ex.recv_buf.to(device_options);
        ex.recv_view.copy_(ex.recv_buf);
      }
    }
  }
}

void
TensorProblem::runComputeWithGhosts(TensorOperatorBase & compute)
{
  const auto & requirements = compute.getInputGhostLayers();
  for (const auto & [buffer_name, ghost] : requirements)
    if (ghost > 0)
      exchangeGhostLayers(buffer_name, ghost);
  compute.computeBuffer();
}

void
TensorProblem::setSolver(std::shared_ptr<TensorSolver> solver,
                         const MooseTensor::Key<CreateTensorSolverAction> &)
{
  if (_solver)
    mooseError("A solver has already been set up.");

  _solver = solver;
}

TensorBufferBase &
TensorProblem::getBufferBase(const std::string & buffer_name)
{
  auto it = _tensor_buffer.find(buffer_name);
  if (it == _tensor_buffer.end())
    mooseError("TensorBuffer '", buffer_name, " does not exist in the system.");
  return *it->second.get();
}

void
TensorProblem::registerGhostLayerRequest(const std::string & buffer_name, unsigned int ghost_layers)
{
  auto & current = _buffer_ghost_layers[buffer_name];
  if (ghost_layers > current)
  {
    current = ghost_layers;
    if (ghost_layers > _max_ghost_layers)
    {
      _max_ghost_layers = ghost_layers;
      updateLocalTensorShape();
    }
  }
}

const torch::Tensor &
TensorProblem::getRawBuffer(const std::string & buffer_name)
{
  return getBufferBase(buffer_name).getRawTensor();
}

const torch::Tensor &
TensorProblem::getRawCPUBuffer(const std::string & buffer_name)
{
  return getBufferBase(buffer_name).getRawCPUTensor();
}

std::vector<int64_t>
TensorProblem::getLocalTensorShape(const std::vector<int64_t> & extra_dims) const
{
  std::vector<int64_t> dims(_dim);
  for (const auto d : make_range(_dim))
    dims[d] = _local_tensor_shape[d];
  dims.insert(dims.end(), extra_dims.begin(), extra_dims.end());
  return dims;
}

TensorProblem &
TensorProblem::cast(MooseObject * moose_object, Problem & problem)
{
  if (auto tensor_problem = dynamic_cast<TensorProblem *>(&problem); tensor_problem)
    return *tensor_problem;
  moose_object->mooseError("Object requires a TensorProblem.");
}
