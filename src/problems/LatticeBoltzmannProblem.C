/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

#include "TensorSolver.h"
#include "TensorOperatorBase.h"
#include "TensorOutput.h"
#include "DomainAction.h"

#include "MarlinUtils.h"
#include "DependencyResolverInterface.h"

registerMooseObject("MarlinApp", LatticeBoltzmannProblem);

InputParameters
LatticeBoltzmannProblem::validParams()
{
  InputParameters params = TensorProblem::validParams();
  params.addParam<TensorInputBufferName>(
      "binary_media",
      "Integer tensor buffer defining solid/fluid regions for complex geometries. "
      "Values: 0 = solid (closed cell, no flow), 1 = fluid (open cell, flow allowed). "
      "Internal solid boundaries must use 'boundary = wall' in boundary conditions. "
      "Domain edge boundaries (top/bottom/left/right/front/back) are specified separately.");
  params.addParam<bool>("enable_slip", false, "Enable slip model");
  // params.addParam<Real>("mfp", 0.0, "Mean free path of the system, (meters)");
  // params.addParam<Real>("dx", 0.0, "Domain resolution, (meters)");
  params.addParam<unsigned int>("substeps", 1, "Number of LBM iterations for every MOOSE timestep");
  params.addParam<Real>("tolerance", 1.0e-15, "LBM convergence tolerance");
  params.addClassDescription("Problem object to enable solving lattice Boltzmann problems");

  return params;
}

LatticeBoltzmannProblem::LatticeBoltzmannProblem(const InputParameters & parameters)
  : TensorProblem(parameters),
    _is_binary_media(isParamValid("binary_media")),
    _enable_slip(getParam<bool>("enable_slip")),
    /*_mfp(getParam<Real>("mfp")),
    _dx(getParam<Real>("dx")),*/
    _lbm_substeps(getParam<unsigned int>("substeps")),
    _tolerance(getParam<Real>("tolerance"))
{
  if (_domain.comm().size() > 1)
    _ghost_radius = 1;

  // fix sizes
  std::vector<int64_t> shape(_domain.getLocalGridSize().begin(), _domain.getLocalGridSize().end());
  if (shape.size() < 3)
    shape.push_back(1);
  for (const auto i : index_range(shape))
  {
    _shape_extended.push_back(shape[i]);
    _shape_extended_to_q.push_back(shape[i]);
  }
}

void
LatticeBoltzmannProblem::init()
{
  TensorProblem::init();

  // dependency resolution of boundary conditions
  DependencyResolverInterface::sort(_bcs);

  // binary mesh if provided
  if (_is_binary_media)
    _binary_media = getBuffer(getParam<TensorInputBufferName>("binary_media"));
  else
    _binary_media = torch::ones(_shape, MooseTensor::intTensorOptions());
}

void
LatticeBoltzmannProblem::execute(const ExecFlagType & exec_type)
{
  if (_convergence_residual < _tolerance)
    return;

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

    // if the binary mesh is updated at initial conditions
    // in the future we need a better way to handle this
    if (_is_binary_media)
      _binary_media = getBuffer(getParam<TensorInputBufferName>("binary_media"));
    else
      _binary_media = torch::ones(_shape, MooseTensor::intTensorOptions());

    executeTensorOutputs(EXEC_INITIAL);
  }

  if (exec_type == EXEC_TIMESTEP_BEGIN && timeStep() > 1)
    for (unsigned substep = 0; substep < _lbm_substeps; substep++)
    {
      // create old state buffers
      advanceState();

      // run solver for streaming
      if (_solver)
        _solver->computeBuffer();

      // run bcs
      for (auto & bc : _bcs)
        bc->realSpaceComputeBuffer();

      // run computes
      for (auto & cmp : _computes)
        cmp->computeBuffer();
      _console << COLOR_WHITE << "Lattice Boltzmann Substep " << substep << ", Residual "
               << _convergence_residual << COLOR_DEFAULT << std::endl;

      _t_total++;
    }

  if (exec_type == EXEC_TIMESTEP_END)
    executeTensorOutputs(EXEC_TIMESTEP_END);

  // mapBuffersToAux();
  FEProblem::execute(exec_type);
}

void
LatticeBoltzmannProblem::addTensorBoundaryCondition(const std::string & compute_type,
                                                    const std::string & name,
                                                    InputParameters & parameters)
{
  addTensorCompute(compute_type, name, parameters, _bcs);
}

void
LatticeBoltzmannProblem::addStencil(const std::string & stencil_name,
                                    const std::string & name,
                                    InputParameters & parameters)
{
  if (_stencil_counter > 0)
    mooseError("Problem object LatticeBoltzmannProblem can only have one stencil");
  // Create the object
  _stencil = _factory.create<LatticeBoltzmannStencilBase>(stencil_name, name, parameters, 0);
  _stencil_counter++;
  logAdd("LatticeBoltzmannStencilBase", name, stencil_name, parameters);

  _shape_extended_to_q.push_back(_stencil->_q);
}

void
LatticeBoltzmannProblem::maskedFillSolids(torch::Tensor & t, const Real & value)
{
  const auto tensor_shape = t.sizes();
  if (_is_binary_media && _binary_media.sum().item<int64_t>() > 0)
  {
    if (t.dim() == _binary_media.dim())
    {
      // 3D
      const auto solid_mask = (_binary_media == value);
      t.masked_fill_(solid_mask, value);
    }
    else
    {
      // 2D and 1D
      const auto solid_mask = (_binary_media == value).unsqueeze(-1).expand(tensor_shape);
      t.masked_fill_(solid_mask, value);
    }
  }
}
