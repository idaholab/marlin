/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MarlinApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "ModulesApp.h"
#include "MooseSyntax.h"
#include "DomainAction.h"
#include "MarlinUtils.h"

#include <cstdlib>

#ifdef LIBMESH_HAVE_HDF5
#include "hdf5.h"
#endif

#ifdef MPIX_CUDA_AWARE_SUPPORT
#include <mpi-ext.h>
#endif

namespace MooseTensor
{
static struct MarlinGlobalSettings
{
  MarlinGlobalSettings()
  {
    const auto env = std::getenv("MARLIN_TORCH_DEVICE");
    if (env)
      _torch_device = std::string(env);
    else
      _torch_device = "";
  }
  std::string _torch_device;
  std::string _floating_precision;
} marlin_global_settings;

std::string
torchDevice()
{
  return marlin_global_settings._torch_device;
}

std::string
precision()
{
  return marlin_global_settings._floating_precision;
}
}

InputParameters
MarlinApp::validParams()
{
  InputParameters params = MooseApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;
  return params;
}

MarlinApp::MarlinApp(const InputParameters & parameters) : MooseApp(parameters)
{
  MarlinApp::registerAll(_factory, _action_factory, _syntax);
  MooseTensor::marlin_global_settings._torch_device =
      std::string(parameters.get<MooseEnum>("compute_device"));
}

MarlinApp::~MarlinApp() {}

void
MarlinApp::setTorchDevice(std::string device, const MooseTensor::Key<DomainAction> &)
{
  MooseTensor::marlin_global_settings._torch_device = device;
}

void
MarlinApp::setTorchDeviceStatic(std::string device, const MooseTensor::Key<MarlinInit> &)
{
  MooseTensor::marlin_global_settings._torch_device = device;
}

void
MarlinApp::setTorchPrecision(std::string precision, const MooseTensor::Key<DomainAction> &)
{
  MooseTensor::marlin_global_settings._floating_precision = precision;
}

void
MarlinApp::registerAll(Factory & f, ActionFactory & af, Syntax & syntax)
{
  ModulesApp::registerAllObjects<MarlinApp>(f, af, syntax);
  Registry::registerObjectsTo(f, {"MarlinApp"});
  Registry::registerActionsTo(af, {"MarlinApp"});

  auto registerDeep = [&](const std::string & base_path, const std::string & task)
  {
    registerMooseObjectTask(task, TensorOperator, false);
    std::string path = base_path;
    // register five levels deep
    for (unsigned int i = 0; i < 5; ++i)
    {
      path += "/*";
      syntax.registerSyntaxType(path, "TensorComputeName");
      registerSyntaxTask("AddTensorComputeAction", path, task);
    }
  };

  // ComputeDevice Action
  registerSyntax("DomainAction", "Domain");

  // LBM Stencil Actions
  registerSyntaxTask("AddLBMStencilAction", "Stencil/*", "add_stencil");
  syntax.registerSyntaxType("Stencil/*", "StencilName");
  registerMooseObjectTask("add_stencil", LBStencil, false);
  addTaskDependency("add_stencil", "add_aux_variable");

  // TensorBuffer Actions
  registerSyntaxTask("AddTensorBufferAction", "TensorBuffers/*", "add_tensor_buffer");
  syntax.registerSyntaxType("TensorBuffers/*", "TensorInputBufferName");
  syntax.registerSyntaxType("TensorBuffers/*", "TensorOutputBufferName");
  registerMooseObjectTask("add_tensor_buffer", TensorBuffer, false);
  addTaskDependency("add_tensor_buffer", "add_stencil");

  // TensorComputes/Initial Actions
  registerDeep("TensorComputes/Initialize", "add_tensor_ic");
  addTaskDependency("add_tensor_ic", "add_tensor_buffer");

  // TensorComputes/Boundary Actions
  registerSyntaxTask("AddLBMBCAction", "TensorComputes/Boundary/*", "add_tensor_bc");
  syntax.registerSyntaxType("TensorComputes/Boundary/*", "TensorComputeName");
  registerMooseObjectTask("add_tensor_bc", TensorOperator, false);
  addTaskDependency("add_tensor_bc", "add_tensor_buffer");

  // TensorComputes/Solve Action
  registerDeep("TensorComputes/Solve", "add_tensor_compute");
  addTaskDependency("add_tensor_compute", "add_tensor_ic");

  // TensorComputes/Postprocess Action
  registerDeep("TensorComputes/Postprocess", "add_tensor_postprocessor");
  addTaskDependency("add_tensor_postprocessor", "add_tensor_compute");

  registerSyntaxTask("EmptyAction", "TensorComputes", "no_action"); // placeholder

  // TensorOutputs Action
  registerSyntaxTask("AddTensorOutputAction", "TensorOutputs/*", "add_tensor_output");
  syntax.registerSyntaxType("TensorOutputs/*", "TensorOutputName");
  registerMooseObjectTask("add_tensor_output", TensorOutput, false);
  addTaskDependency("add_tensor_output", "add_tensor_postprocessor");

  // Create TensorSolver
  registerSyntaxTask("CreateTensorSolverAction", "TensorSolver", "create_tensor_solver");
  registerMooseObjectTask("create_tensor_solver", TensorSolver, false);
  addTaskDependency("create_tensor_solver", "add_tensor_output");

  // Add predictors to the solver
  registerSyntaxTask(
      "AddTensorPredictorAction", "TensorSolver/Predictors/*", "add_tensor_predictor");
  syntax.registerSyntaxType("TensorSolver/Predictors/*", "TensorPredictorName");
  registerMooseObjectTask("add_tensor_predictor", TensorPredictor, false);
  addTaskDependency("add_tensor_predictor", "create_tensor_solver");

  // Register data file path
  registerAppDataFilePath("marlin");

  // make sure all this gets run before `add_mortar_variable`
  addTaskDependency("add_mortar_variable", "add_tensor_predictor");
}

void
MarlinApp::registerApps()
{
  {
    const std::string doc = "A threadsafe version of libhdf5 ";
#ifdef LIBMESH_HAVE_HDF5
    // Check if the library is thread-safe
    hbool_t is_threadsafe;
    H5is_library_threadsafe(&is_threadsafe);
    addCapability("hdf5_threadsafe", true, doc + "is available.");
#else
  addCapability("hdf5_threadsafe", false, doc + "is not available.");
#endif
  }

  {
    const std::string doc = "A CUDA enabled version of MPI ";
    bool flag = false;
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    flag = MPIX_Query_cuda_support();
#endif
    addCapability("mpi_cuda_aware", flag, doc + (flag ? "is available." : "is not available."));
  }

  registerApp(MarlinApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
extern "C" void
MarlinApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  MarlinApp::registerAll(f, af, s);
}
extern "C" void
MarlinApp__registerApps()
{
  MarlinApp::registerApps();
}
