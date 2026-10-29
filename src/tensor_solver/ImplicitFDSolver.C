/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ImplicitFDSolver.h"
#include "TensorProblem.h"
#include "TensorBufferBase.h"
#include "DomainAction.h"
#include "MarlinUtils.h"
#include "ComputeGroup.h"
#include "TransientBase.h"
#include "TensorSolveIterationAdaptiveDT.h"
#include "TensorRobinBC.h"
#include "FiniteDifferenceBiharmonic.h"
#include "FiniteDifferenceDivGrad.h"

#include "PetscSupport.h"

#include "libmesh/int_range.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>

// the PETSc backend on CUDA needs PETSc and torch both built with CUDA
#if defined(__has_include)
#if __has_include(<ATen/cuda/CUDAConfig.h>)
#define MARLIN_TORCH_HAS_CUDA 1
#endif
#endif
#if PetscDefined(HAVE_CUDA) && defined(MARLIN_TORCH_HAS_CUDA)
#define MARLIN_PETSC_CUDA 1
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <petscdevice_cuda.h>
#else
#define MARLIN_PETSC_CUDA 0
#endif

registerMooseObject("MarlinApp", ImplicitFDSolver);

namespace
{
/// a PETSc option of the nonlinear solve (prefix nonlinear_) rather than of the field solves
bool
isNonlinearOption(const std::string & name)
{
  const auto lower = MooseUtils::toLower(name);
  return lower.rfind("-snes", 0) == 0 || lower.rfind("-npc_", 0) == 0;
}
}

InputParameters
ImplicitFDSolver::validParams()
{
  InputParameters params = TensorSolver::validParams();
  params.addClassDescription(
      "Implicit solver for fields whose linear terms are finite-difference operators, solving "
      "sum_i A_i u = sum_i r_i + f for each field with a Krylov method.");

  params.addRequiredParam<std::vector<TensorOutputBufferName>>("buffer", "The solved fields");
  params.addRequiredParam<std::vector<std::vector<TensorComputeName>>>(
      "operators",
      "For each field (separated by ';'), the LinearFDOperator computes whose sum is the system "
      "operator");
  params.addParam<std::vector<TensorInputBufferName>>(
      "source", {}, "Explicit source term f for each field (empty for no sources)");

  MooseEnum linear_solver("AUTO PETSC TORCH_PCG", "AUTO");
  params.addParam<MooseEnum>("linear_solver",
                             linear_solver,
                             "Linear solver backend. AUTO selects TORCH_PCG on MPS and PETSC on "
                             "all other devices.");
  params.addRangeCheckedParam<Real>(
      "l_tol", 1e-8, "l_tol > 0", "Relative tolerance on the unpreconditioned residual norm");
  params.addRangeCheckedParam<unsigned int>(
      "l_max_its", 10000, "l_max_its > 0", "Maximum number of linear iterations");
  MooseEnum smoother("CHEBYSHEV DAMPED_JACOBI", "CHEBYSHEV");
  params.addParam<MooseEnum>(
      "smoother",
      smoother,
      "Multigrid level smoother of the PETSC linear solver. CHEBYSHEV is PCMG's default, with "
      "SOR on the coarse levels on CPU and Jacobi on the fine level and on CUDA; DAMPED_JACOBI "
      "is Jacobi weighted by the Gershgorin bound of each level operator.");

  params.addParam<bool>(
      "nonlinear_solve",
      false,
      "Iterate each step to convergence of the coupled system sum_i A_i u = sum_i r_i + f(U) of "
      "all fields, with the explicit computes and buffer-valued boundary data re-evaluated at the "
      "iterate and at t + dt (backward Euler). The field solves precondition the iteration. "
      "Without it the explicit terms are evaluated once per step, at t (lagged).");
  params.addRangeCheckedParam<Real>("nl_rel_tol",
                                    1e-8,
                                    "nl_rel_tol >= 0 & nl_rel_tol < 1",
                                    "Relative tolerance on the coupled residual 2-norm of the "
                                    "nonlinear solve, relative to its initial value");
  params.addRangeCheckedParam<Real>("nl_abs_tol",
                                    1e-12,
                                    "nl_abs_tol >= 0",
                                    "Absolute tolerance on the coupled residual 2-norm of the "
                                    "nonlinear solve");
  params.addRangeCheckedParam<unsigned int>(
      "nl_max_its", 50, "nl_max_its > 0", "Maximum number of nonlinear iterations");
  MooseEnum nonlinear_solver("AUTO PETSC TORCH", "AUTO");
  params.addParam<MooseEnum>(
      "nonlinear_solver",
      nonlinear_solver,
      "Nonlinear solver backend of nonlinear_solve: PETSC is PETSc's SNES (methods by "
      "petsc_options, e.g. Anderson), TORCH the plain Picard iteration. Either works with either "
      "linear solver; with PETSc's vectors on the host and the field solves on a GPU (MPS, or "
      "CUDA without a CUDA PETSc), every iterate is copied between them. AUTO selects PETSC with "
      "linear_solver = PETSC or when nonlinear options (-snes_*, -npc_*) are given, TORCH "
      "otherwise.");

  // PETSc options: options starting with -snes or -npc_ go to the nonlinear solve (prefix
  // nonlinear_, nonlinear_solver = PETSC), all others to each field's solve (prefix <buffer>_,
  // linear_solver = PETSC)
  params += Moose::PetscSupport::getPetscValidParams();
  params.suppressParameter<MooseEnum>("solve_type");
  params.suppressParameter<MooseEnum>("mffd_type");
  // one solve per time step; a failed solve is retried with a smaller dt
  params.suppressParameter<unsigned int>("substeps");

  return params;
}

ImplicitFDSolver::ImplicitFDSolver(const InputParameters & parameters)
  : TensorSolver(parameters),
    IterativeTensorSolverInterface(),
    _l_tol(getParam<Real>("l_tol")),
    _l_max_its(getParam<unsigned int>("l_max_its")),
    _damped_jacobi(getParam<MooseEnum>("smoother") == "DAMPED_JACOBI"),
    _nonlinear_solve(getParam<bool>("nonlinear_solve")),
    _nl_rel_tol(getParam<Real>("nl_rel_tol")),
    _nl_abs_tol(getParam<Real>("nl_abs_tol")),
    _nl_max_its(getParam<unsigned int>("nl_max_its"))
{
  // backend selection
  const auto device = MooseTensor::floatTensorOptions().device();
  const auto & linear_solver = getParam<MooseEnum>("linear_solver");
  if (linear_solver == "AUTO")
  {
    _linear_solver = device.is_mps() ? LinearSolver::TORCH_PCG : LinearSolver::PETSC;
    mooseInfo("Using the ",
              _linear_solver == LinearSolver::PETSC ? "PETSC" : "TORCH_PCG",
              " linear solver on device '",
              device.str(),
              "'.");
  }
  else
    _linear_solver = linear_solver == "PETSC" ? LinearSolver::PETSC : LinearSolver::TORCH_PCG;

  if (_linear_solver == LinearSolver::PETSC)
  {
    if (device.is_mps())
      paramError("linear_solver",
                 "The PETSC linear solver does not run on MPS. Use linear_solver = TORCH_PCG.");
    if (torch::typeMetaToScalarType(MooseTensor::floatTensorOptions().dtype()) != torch::kDouble)
      paramError("linear_solver",
                 "The PETSC linear solver requires double precision. Use "
                 "linear_solver = TORCH_PCG for single precision.");
    if (device.is_cuda() && !MARLIN_PETSC_CUDA)
      paramError("linear_solver",
                 "The PETSC linear solver on CUDA requires PETSc and libtorch built with CUDA. Use "
                 "linear_solver = TORCH_PCG.");
  }
  else if (isParamSetByUser("smoother"))
    paramError("smoother", "This parameter is not used by linear_solver = TORCH_PCG.");

  // the PETSc options given (an empty list sets none), split into those of the field solves and
  // those of the nonlinear solve
  std::vector<std::string> field_options, nonlinear_options;
  {
    std::vector<std::string> option_names;
    for (const auto & flag : getParam<MultiMooseEnum>("petsc_options"))
      option_names.push_back(MooseUtils::toLower(flag.rawName()));
    for (const auto & name : getParam<MultiMooseEnum>("petsc_options_iname"))
      option_names.push_back(MooseUtils::toLower(name.rawName()));
    for (const auto & name : option_names)
      (isNonlinearOption(name) ? nonlinear_options : field_options).push_back(name);
  }
  if (!field_options.empty() && _linear_solver != LinearSolver::PETSC)
    paramError("petsc_options",
               "The option '",
               field_options.front(),
               "' configures the field solves, which use PETSc only with linear_solver = PETSC.");

  if (!_nonlinear_solve)
  {
    for (const auto & name : {"nl_rel_tol", "nl_abs_tol", "nl_max_its", "nonlinear_solver"})
      if (isParamSetByUser(name))
        paramError(name, "This parameter is used only with nonlinear_solve = true.");
    if (!nonlinear_options.empty())
      paramError("petsc_options",
                 "The option '",
                 nonlinear_options.front(),
                 "' configures the nonlinear solve, which is used only with nonlinear_solve = "
                 "true.");
  }

  // PETSc's SNES drives the nonlinear iteration with either linear solver; AUTO picks it with
  // the PETSc linear solver or when its options are given, the torch Picard loop otherwise
  const auto & nonlinear_solver = getParam<MooseEnum>("nonlinear_solver");
  if (nonlinear_solver == "AUTO")
    _nonlinear_solver = _linear_solver == LinearSolver::PETSC || !nonlinear_options.empty()
                            ? NonlinearSolver::PETSC
                            : NonlinearSolver::TORCH;
  else
    _nonlinear_solver =
        nonlinear_solver == "PETSC" ? NonlinearSolver::PETSC : NonlinearSolver::TORCH;
  if (_nonlinear_solver == NonlinearSolver::TORCH && !nonlinear_options.empty())
    paramError("petsc_options",
               "The option '",
               nonlinear_options.front(),
               "' configures PETSc's nonlinear solve, which is not used with nonlinear_solver = "
               "TORCH.");

  // the operators act on ghosted real space buffers
  if (!_domain.isRealSpaceMode())
    mooseError("ImplicitFDSolver requires Domain/parallel_mode = REAL_SPACE.");
  const auto & periodic = _domain.getPeriodicDirections();
  const bool has_boundaries = std::any_of(
      periodic.begin(), periodic.begin() + _dim, [](bool is_periodic) { return !is_periodic; });

  const auto & buffers = getParam<std::vector<TensorOutputBufferName>>("buffer");
  const auto & operators = getParam<std::vector<std::vector<TensorComputeName>>>("operators");
  const auto & sources = getParam<std::vector<TensorInputBufferName>>("source");
  if (operators.size() != buffers.size())
    paramError("operators", "Must have one ';'-separated list of operators per entry in 'buffer'.");
  if (!sources.empty() && sources.size() != buffers.size())
    paramError("source", "Must be empty or have one entry per entry in 'buffer'.");

  // PETSc matches option names case-insensitively, so such fields would share a prefix
  for (const auto i : index_range(buffers))
  {
    if (const auto name = MooseUtils::toLower(buffers[i]);
        name == "nonlinear" || MooseUtils::startsWith(name, "nonlinear_"))
      paramError("buffer",
                 "A field named '",
                 buffers[i],
                 "' would share the PETSc options prefix of the nonlinear solve.");
    for (const auto j : make_range(i))
      if (MooseUtils::toLower(buffers[i]) == MooseUtils::toLower(buffers[j]))
        paramError("buffer",
                   "Fields '",
                   buffers[j],
                   "' and '",
                   buffers[i],
                   "' differ only in case, so their PETSc options prefixes would collide.");
  }

  _fields.reserve(buffers.size());
  for (const auto i : make_range(buffers.size()))
  {
    Field field{buffers[i],
                getOutputBufferByName(buffers[i]),
                _tensor_problem.getBufferBase(buffers[i]),
                {},
                {},
                {},
                sources.empty() ? nullptr : &getBufferBase(sources[i]),
                false,
                0.0,
                nullptr,
                {},
                {},
                {}};

    if (operators[i].empty())
      paramError("operators", "No operators given for field '", buffers[i], "'.");

    // sum_i A_i is SPD if one term is SPD and the others are SPD or SPSD; it is
    // SPSD with the constants as nullspace if all terms are SPSD
    bool definite = false;
    for (const auto & op_name : operators[i])
    {
      auto & op = _tensor_problem.getCompute(op_name);
      const auto * fd_op = dynamic_cast<const LinearFDOperator *>(&op);
      if (!fd_op)
        paramError("operators", "Compute '", op_name, "' is not a LinearFDOperator.");
      if (fd_op->inputBufferName() != buffers[i])
        paramError("operators",
                   "Operator '",
                   op_name,
                   "' acts on '",
                   fd_op->inputBufferName(),
                   "' rather than on the solved field '",
                   buffers[i],
                   "'.");

      switch (fd_op->definiteness())
      {
        case LinearFDOperator::Definiteness::POSITIVE_DEFINITE:
          definite = true;
          break;
        case LinearFDOperator::Definiteness::POSITIVE_SEMIDEFINITE:
          break;
        case LinearFDOperator::Definiteness::OTHER:
          paramError("operators",
                     "Operator '",
                     op_name,
                     "' is not symmetric positive (semi)definite, which the conjugate gradient "
                     "solve requires. For -D*Laplacian(u) use FiniteDifferenceLaplacian with "
                     "factor = -D.");
      }

      // the 5-point biharmonic applies its composed stencil to the ghosts of u directly, without
      // the face pairs the 3-point one imposes on the intermediate Laplacian
      // (FiniteDifferenceBiharmonic::setFaceReflections); not supported next to a face yet
      if (has_boundaries && dynamic_cast<const FiniteDifferenceBiharmonic *>(fd_op) &&
          PetscFD::stencilWidth(fd_op->stencilAt()) > 2)
        paramError("operators",
                   "Operator '",
                   op_name,
                   "' is a 5-point FiniteDifferenceBiharmonic, which ImplicitFDSolver supports on "
                   "periodic domains only. Use stencil_width = 3.");

      field._operators.push_back(&op);
      field._fd_operators.push_back(fd_op);
      field._operator_outputs.push_back(
          &_tensor_problem.getBufferBase(op.getParam<TensorOutputBufferName>("buffer")));
    }

    // A boundary face's a*u + b*du/dn = c sets the ghost next to the face cell u0 to coeff*u0 +
    // const, which changes only the diagonal of the 3-point operator, so it stays symmetric. With
    // a/(a/2 + b/dx) > 0 (coeff < 1, e.g. Dirichlet) it becomes definite, with a = 0
    // (Neumann, coeff = 1) it keeps the constants as nullspace, and otherwise it can be
    // indefinite. The 5-point stencil's second ghost uses spacing 3dx, so its operator is
    // symmetric only where both ghost coefficients agree: for Dirichlet and Neumann.
    field._face_bcs.assign(2 * _dim, nullptr);
    for (const auto face : make_range(2 * _dim))
    {
      const auto d = face / 2;
      if (periodic[d])
        continue;
      // without a boundary condition no operator reads ghosts of u (checkBoundaryConditions)
      const auto * bc = _tensor_problem.getBoundaryCondition(buffers[i], face);
      if (!bc)
        continue;
      const auto * face_bc = dynamic_cast<const TensorRobinBC *>(bc);
      if (!face_bc)
        paramError("buffer",
                   "Boundary condition '",
                   bc->name(),
                   "' of field '",
                   buffers[i],
                   "' is not a TensorRobinBC, which ImplicitFDSolver requires.");
      field._face_bcs[face] = face_bc;

      // with a*b >= 0 the ghost term a/(a/2 + b/dx) is positive for every dx; with a*b < 0 its
      // sign, and so the definiteness of the operator, would depend on the grid
      const auto a = face_bc->a(), b = face_bc->b();
      if (a * b < 0.0)
        paramError("buffer",
                   "Boundary condition '",
                   bc->name(),
                   "' of field '",
                   buffers[i],
                   "' has a*b < 0, for which the ghost term a/(a/2 + b/dx) is negative on fine "
                   "enough grids and the operator may be indefinite.");
      if (a != 0.0)
        definite = true;

      // The biharmonic sets the ghosts of w = Laplacian(u) to -w0 (Dirichlet) or w0 (Neumann):
      // the face pairs are (u = c, Laplacian(u) = 0) and (du/dn = c, dLaplacian(u)/dn = 0),
      // and the operator is the square of the folded Laplacian. With a Robin face it is not.
      if (a != 0.0 && b != 0.0)
        for (const auto * fd_op : field._fd_operators)
          if (dynamic_cast<const FiniteDifferenceBiharmonic *>(fd_op))
            paramError("buffer",
                       "Boundary condition '",
                       bc->name(),
                       "' of field '",
                       buffers[i],
                       "' is a Robin condition (a and b nonzero), which ImplicitFDSolver does not "
                       "support with a FiniteDifferenceBiharmonic. Use Dirichlet or Neumann "
                       "faces.");

      if (a != 0.0 && b != 0.0)
        for (const auto * fd_op : field._fd_operators)
          for (const auto & offset : fd_op->stencilOffsets())
            if (std::abs(offset[d]) > 1)
              paramError("buffer",
                         "Boundary condition '",
                         bc->name(),
                         "' of field '",
                         buffers[i],
                         "' is a Robin condition (a and b nonzero), which ImplicitFDSolver "
                         "supports with the 3-point stencil only: the 5-point operator with it "
                         "is not symmetric.");
    }
    field._singular = !definite;

    _fields.push_back(std::move(field));
  }
}

void
ImplicitFDSolver::findExplicitComputes()
{
  // the root compute updates the explicit terms; the field operators are applied by the
  // solve itself, so they are left out (an automatic root compute contains every compute)
  std::set<const TensorOperatorBase *> operators;
  for (const auto & field : _fields)
    operators.insert(field._operators.begin(), field._operators.end());
  const auto * group = dynamic_cast<const ComputeGroup *>(_compute.get());
  if (group)
    for (const auto & cmp : group->getComputes())
      if (!operators.count(cmp.get()))
        _explicit_computes.push_back(cmp.get());

  // a group without field operators (or a single explicit compute) runs as a whole, keeping
  // its JIT and debug checks
  if (group ? _explicit_computes.size() == group->getComputes().size()
            : !operators.count(_compute.get()))
    _explicit_computes = {_compute.get()};
  _explicit_computes_found = true;
}

void
ImplicitFDSolver::substep()
{
  // update the explicit terms (sources); the nonlinear solve evaluates them at its iterates
  if (!_explicit_computes_found)
    findExplicitComputes();
  if (!_nonlinear_solve)
  {
    for (auto * cmp : _explicit_computes)
      cmp->realSpaceComputeBuffer();
    forwardBuffers();
  }

  _iterations = 0;
  _is_converged = true;
  for (auto & field : _fields)
  {
    // sum_i r_i, evaluated while u still holds u^n (each r_i reads its own field only)
    field._rhs_old = torch::zeros_like(field._u_base.ownedView(), torch::MemoryFormat::Contiguous);
    for (const auto * fd_op : field._fd_operators)
      if (const auto r = fd_op->rhsContribution(); r.defined())
        field._rhs_old += r;

    // u is Krylov scratch below; give it its own storage (advanceState() keeps a
    // shallow copy in its old states)
    field._u_old = field._u;
    field._u = field._u.clone();
  }

  if (_nonlinear_solve)
  {
    nonlinearSubstep();
    return;
  }

  // boundary functions are evaluated at the time level of the unknown, t + dt; buffer values
  // are those the explicit computes above wrote, at t
  _tensor_problem.beginFixedBoundaryData(_sub_dt);
  for (auto & field : _fields)
  {
    // warm start from u_old
    torch::Tensor x = field._u_base.ownedView().clone(torch::MemoryFormat::Contiguous);
    torch::Tensor b = fieldRhs(field);

    const auto result = solveField(field, b, x);
    _iterations += result.iterations;
    if (!result.failure.empty())
    {
      failSubstep(result.failure);
      _tensor_problem.endFixedBoundaryData();
      return;
    }

    // write x into u and apply the operators to it once more, so that their output buffers
    // hold A_i u (with the boundary values) rather than A_i of the last Krylov vector
    applyOperator(field, x);
    _tensor_problem.exchangeGhostLayers(field._name,
                                        _tensor_problem.getBufferGhostLayers(field._name));
  }
  _tensor_problem.endFixedBoundaryData();
}

torch::Tensor
ImplicitFDSolver::fieldRhs(Field & field)
{
  // b = sum_i r_i + f
  torch::Tensor b = field._rhs_old.clone();
  if (field._source)
    b += field._source->ownedView();

  // with boundary faces the operator action is affine, sum_i A_i x + g with g from the boundary
  // values: solve the linear part with b - g (g = sum_i A_i 0, for the current values)
  if (hasBoundaryFaces(field))
    b -= applyOperator(field, torch::zeros_like(b));

  if (field._singular)
  {
    // remove only a round-off mean: pairwise summation errs by ~log2(N) eps sum|b|
    // (Higham, ch. 4); 16 covers rounding in b and the cross-rank reduction. The round-off
    // left in a conserved source (e.g. the mean of a decaying c) is that of its largest
    // magnitude, so the scale is the largest sum|b| so far, not the current one
    const auto n_cells = static_cast<Real>(_domain.getNumberOfCells());
    const Real eps = b.scalar_type() == torch::kFloat ? std::numeric_limits<float>::epsilon()
                                                      : std::numeric_limits<double>::epsilon();
    const Real b_sum = _domain.allreduceSum(b.sum()).cpu().item<double>();
    field._max_b_abs_sum =
        std::max(field._max_b_abs_sum, _domain.allreduceSum(b.abs().sum()).cpu().item<double>());
    if (std::abs(b_sum) > 16.0 * std::log2(n_cells) * eps * field._max_b_abs_sum)
      mooseError("The operator of field '",
                 field._name,
                 "' is singular (its nullspace is the constants), so the right-hand side must "
                 "have zero mean, but its mean is ",
                 b_sum / n_cells,
                 ".");
    b -= b_sum / n_cells;
  }
  return b;
}

ImplicitFDSolver::SolveResult
ImplicitFDSolver::solveField(Field & field, torch::Tensor & b, torch::Tensor & x)
{
  // with A 1 = lambda 1 (A symmetric), the sum of the equations fixes the mean of the exact
  // solution, mean(x) = mean(b) / lambda: the Krylov error in the constant mode is set to zero
  // after the solve (e.g. mass of a conserved field on periodic or Neumann faces)
  const Real lambda = field._singular ? 0.0 : constantEigenvalue(field);
  const Real b_mean = lambda > 0.0 ? _domain.allreduceSum(b.sum()).cpu().item<double>() /
                                         static_cast<Real>(_domain.getNumberOfCells())
                                   : 0.0;

  SolveResult result;
  const Real b_norm = std::sqrt(_domain.allreduceSum((b * b).sum()).cpu().item<double>());
  if (b_norm == 0.0)
    // the zero-mean solution of A x = 0 is x = 0
    x.zero_();
  else if (_linear_solver == LinearSolver::PETSC)
    result = solvePetsc(field, b, x);
  else
    result = solveTorchPCG(field, b, b_norm, x);
  if (!result.failure.empty())
    return result;

  if (field._singular)
    x -= _domain.allreduceSum(x.sum()) / static_cast<Real>(_domain.getNumberOfCells());
  else if (lambda > 0.0)
    x += b_mean / lambda -
         _domain.allreduceSum(x.sum()) / static_cast<Real>(_domain.getNumberOfCells());
  return result;
}

void
ImplicitFDSolver::failSubstep(const std::string & failure)
{
  // only a TensorSolveIterationAdaptiveDT time stepper cuts dt on a failed solve
  const auto * transient = dynamic_cast<const TransientBase *>(_app.getExecutioner());
  if (!transient ||
      !dynamic_cast<const TensorSolveIterationAdaptiveDT *>(transient->getTimeStepper()))
    mooseError(failure,
               " Use the TensorSolveIterationAdaptiveDT time stepper to retry the step with a "
               "smaller dt.");

  _console << failure << std::endl;
  // retry from the previous state of every field
  for (auto & field : _fields)
    field._u = field._u_old;
  _is_converged = false;
}

torch::Tensor
ImplicitFDSolver::applyOperator(Field & field, const torch::Tensor & x)
{
  field._u_base.ownedView().copy_(x);

  torch::Tensor y;
  for (const auto i : index_range(field._operators))
  {
    _tensor_problem.runComputeWithGhosts(*field._operators[i]);
    const auto output = field._operator_outputs[i]->ownedView();
    y = y.defined() ? y + output : output.clone();
  }
  return y;
}

bool
ImplicitFDSolver::hasBoundaryFaces(const Field & field)
{
  return std::any_of(
      field._face_bcs.begin(), field._face_bcs.end(), [](const auto * face_bc) { return face_bc; });
}

torch::Tensor
ImplicitFDSolver::applyLinearOperator(Field & field, const torch::Tensor & x)
{
  // the operator with zero boundary data is A x itself; (A x + g) - g would leave a round-off
  // error of eps |g|, a floor for the solve of a small right-hand side (a nonlinear correction)
  // restored on every exit (an exception from the operator), to the caller's value if nested
  struct HomogeneousScope
  {
    TensorProblem & problem;
    const bool previous = problem.homogeneousBoundaryData();
    ~HomogeneousScope() { problem.setHomogeneousBoundaryData(previous); }
  } scope{_tensor_problem};
  _tensor_problem.setHomogeneousBoundaryData(true);
  return applyOperator(field, x);
}

namespace
{
/// the stencil entries of all operators at the given spacing (empty: the domain's)
std::vector<FDStencilEntry>
fieldStencil(const std::vector<const LinearFDOperator *> & fd_operators,
             const std::vector<Real> & grid_spacing = {})
{
  std::vector<FDStencilEntry> stencil;
  for (const auto * fd_op : fd_operators)
  {
    const auto entries = grid_spacing.empty() ? fd_op->stencilAt() : fd_op->stencilAt(grid_spacing);
    stencil.insert(stencil.end(), entries.begin(), entries.end());
  }
  return stencil;
}

/// the homogeneous ghost relations of a field's boundary faces
PetscFD::FaceGhosts
faceGhosts(const std::vector<const TensorRobinBC *> & face_bcs)
{
  PetscFD::FaceGhosts faces;
  for (const auto face : index_range(face_bcs))
    if (face_bcs[face])
      faces.ab[face] = {face_bcs[face]->a(), face_bcs[face]->b()};
  return faces;
}
}

void
ImplicitFDSolver::checkPositiveDiagonal(const Field & field, const torch::Tensor & diag) const
{
  // a zero diagonal (e.g. a steady FiniteDifferenceDivGrad with D = 0 on every face of a cell)
  // is a cell decoupled from the rest: the operator is singular there
  if (_domain.allreduceSum((diag <= 0).sum()).cpu().item<int64_t>() > 0)
    mooseError("The operator of field '",
               field._name,
               "' has a nonpositive diagonal in some cells, so it is not positive definite there "
               "(e.g. a FiniteDifferenceDivGrad coefficient that is zero on every face of a cell, "
               "without a time derivative).");
}

Real
ImplicitFDSolver::constantEigenvalue(Field & field)
{
  // a Dirichlet or Robin face (a != 0) gives the constants a profile next to the face
  for (const auto * face_bc : field._face_bcs)
    if (face_bc && face_bc->a() != 0.0)
      return 0.0;

  // A 1 is exact up to the cancellation of the stencil weights, of the size of the diagonal
  const auto ones = torch::ones_like(field._u_base.ownedView(), torch::MemoryFormat::Contiguous);
  const auto y = applyLinearOperator(field, ones);
  const auto diag = jacobiDiagonal(field);
  const auto n_cells = static_cast<Real>(_domain.getNumberOfCells());
  const Real lambda = _domain.allreduceSum(y.sum()).cpu().item<double>() / n_cells;
  const Real deviation = std::sqrt(
      _domain.allreduceSum(((y - lambda) * (y - lambda)).sum()).cpu().item<double>() / n_cells);
  const Real diag_rms =
      std::sqrt(_domain.allreduceSum((diag * diag).sum()).cpu().item<double>() / n_cells);
  const Real eps = y.scalar_type() == torch::kFloat ? std::numeric_limits<float>::epsilon()
                                                    : std::numeric_limits<double>::epsilon();
  return deviation <= 64.0 * eps * diag_rms && lambda > 64.0 * eps * diag_rms ? lambda : 0.0;
}

torch::Tensor
ImplicitFDSolver::jacobiDiagonal(const Field & field) const
{
  // the diagonal of the folded operator the PETSc levels assemble, so in a cell next to a
  // boundary face it includes the entries whose ghost depends on the cell itself
  std::array<int64_t, 3> begin, end, size;
  _domain.getLocalBounds(_domain.comm().rank(), begin, end);
  for (const auto d : make_range(3))
    size[d] = end[d] - begin[d];
  const auto & dx = _domain.getGridSpacing();
  std::vector<Real> h(_dim);
  for (const auto d : make_range(_dim))
    h[d] = dx(d);
  return PetscFD::foldedDiagonal(fieldStencil(field._fd_operators),
                                 _dim,
                                 _domain.getGridSize(),
                                 _domain.getPeriodicDirections(),
                                 faceGhosts(field._face_bcs),
                                 h,
                                 begin,
                                 size,
                                 MooseTensor::floatTensorOptions());
}

ImplicitFDSolver::SolveResult
ImplicitFDSolver::solveTorchPCG(Field & field,
                                const torch::Tensor & b,
                                Real b_norm,
                                torch::Tensor & x)
{
  // Jacobi preconditioner
  const auto diag = jacobiDiagonal(field);
  checkPositiveDiagonal(field, diag);
  const auto inv_diag = diag.reciprocal();
  const auto A = [&](const torch::Tensor & v) { return applyLinearOperator(field, v); };
  const auto M = [&inv_diag](const torch::Tensor & r) { return r * inv_diag; };
  const auto reduce_sum = [this](const torch::Tensor & t) { return _domain.allreduceSum(t); };

  const auto [x_new, iterations, res_norm] =
      MooseTensor::conjugateGradientSolve(A, b, x, _l_tol, _l_max_its, M, reduce_sum);

  // written so that a NaN residual counts as not converged
  if (!(res_norm <= _l_tol * b_norm))
  {
    std::ostringstream failure;
    failure << "The TORCH_PCG solve of field '" << field._name << "' did not converge in "
            << _l_max_its << " iterations (relative residual " << res_norm / b_norm
            << ", l_tol = " << _l_tol << ").";
    return {iterations, failure.str()};
  }

  x = x_new;
  return {iterations, ""};
}

ImplicitFDSolver::PetscSolve::~PetscSolve()
{
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&b));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&x));
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&ksp));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&fine));
  for (auto & B : coarse)
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&B));
  PetscCallAbort(PETSC_COMM_WORLD, MatNullSpaceDestroy(&nullspace));
  hierarchy.reset();
  if (comm != MPI_COMM_NULL)
    MPI_Comm_free(&comm);
}

namespace
{
bool
sameStencil(const std::vector<FDStencilEntry> & a, const std::vector<FDStencilEntry> & b)
{
  if (a.size() != b.size())
    return false;
  for (const auto i : index_range(a))
  {
    if (a[i].offset != b[i].offset || a[i].coefficient != b[i].coefficient ||
        a[i].field.defined() != b[i].field.defined())
      return false;
    if (a[i].field.defined() && !torch::equal(a[i].field, b[i].field))
      return false;
  }
  return true;
}

#if MARLIN_PETSC_CUDA
/// makes torch enqueue on PETSc's current CUDA stream while a shell callback runs
class PetscStreamGuard
{
public:
  PetscStreamGuard()
  {
    cudaStream_t petsc_stream;
    PetscCallAbort(PETSC_COMM_SELF, PetscGetCurrentCUDAStream(&petsc_stream));
    _guard = std::make_unique<c10::cuda::CUDAStreamGuard>(
        c10::cuda::getStreamFromExternal(petsc_stream, c10::cuda::current_device()));
  }

private:
  std::unique_ptr<c10::cuda::CUDAStreamGuard> _guard;
};
#endif

/// options prefix of a field's KSP
std::string
petscPrefix(const std::string & name)
{
  return name + "_";
}

bool
onCuda()
{
  return MooseTensor::floatTensorOptions().device().is_cuda();
}
}

void
ImplicitFDSolver::insertPetscOptions(const std::string & prefix, bool nonlinear)
{
  // the options of the nonlinear solve or those of the field solves
  auto flags = getParam<MultiMooseEnum>("petsc_options");
  for (const auto & flag : getParam<MultiMooseEnum>("petsc_options"))
    if (isNonlinearOption(flag.rawName()) != nonlinear)
      flags.eraseSetValue(flag.rawName());
  std::vector<std::pair<MooseEnumItem, std::string>> pairs;
  for (const auto & pair :
       getParam<MooseEnumItem, std::string>("petsc_options_iname", "petsc_options_value"))
    if (isNonlinearOption(pair.first.rawName()) == nonlinear)
      pairs.push_back(pair);

  Moose::PetscSupport::PetscOptions po;
  Moose::PetscSupport::addPetscFlagsToPetscOptions(flags, prefix, *this, po);
  Moose::PetscSupport::addPetscPairsToPetscOptions(pairs, _dim, prefix, *this, po);

  // global database, so the PCMG level KSPs (prefix <buffer>_mg_levels_*) see them too
  for (const auto & flag : po.flags)
    Moose::PetscSupport::setSinglePetscOption(flag.rawName());
  for (const auto & [name, value] : po.pairs)
    Moose::PetscSupport::setSinglePetscOption(name, value);
  // command line options win
  Moose::PetscSupport::addPetscOptionsFromCommandline();
}

void
ImplicitFDSolver::setupPetsc(Field & field)
{
  field._petsc = std::make_unique<PetscSolve>();
  auto & ps = *field._petsc;
  ps.solver = this;
  ps.field = &field;

  const auto partition = PetscFD::gatherPartition(_domain);
  ps.comm = PetscFD::dmdaComm(partition);
  const auto comm = ps.comm;

  const auto stencil = fieldStencil(field._fd_operators);
  const auto width = PetscFD::stencilWidth(stencil);
  const auto levels = PetscFD::countLevels(partition, width);
  ps.hierarchy = std::make_unique<PetscFD::Hierarchy>();
  PetscFD::buildHierarchy(partition,
                          comm,
                          levels,
                          width,
                          PetscFD::needsBoxStencil(stencil),
                          faceGhosts(field._face_bcs),
                          *ps.hierarchy);
  auto & hierarchy = *ps.hierarchy;

  const bool cuda = onCuda();
#if MARLIN_PETSC_CUDA
  if (cuda)
  {
    for (auto & da : hierarchy.dms)
    {
      PetscCallAbort(comm, DMSetVecType(da, VECCUDA));
      PetscCallAbort(comm, DMSetMatType(da, MATAIJCUSPARSE));
    }
    for (auto & P : hierarchy.interpolations)
      PetscCallAbort(comm, MatConvert(P, MATAIJCUSPARSE, MAT_INPLACE_MATRIX, &P));
  }
#endif

  // fine level: the operators' own kernels behind a shell
  const PetscInt local = field._u_base.ownedView().numel();
  const PetscInt global = _domain.getNumberOfCells();
  PetscCallAbort(comm, MatCreateShell(comm, local, local, global, global, &ps, &ps.fine));
  PetscCallAbort(comm, MatShellSetOperation(ps.fine, MATOP_MULT, (void (*)(void))shellMult));
  PetscCallAbort(comm,
                 MatShellSetOperation(ps.fine, MATOP_GET_DIAGONAL, (void (*)(void))shellDiagonal));
#if MARLIN_PETSC_CUDA
  if (cuda)
    PetscCallAbort(comm, MatShellSetVecType(ps.fine, VECCUDA));
#endif

  if (field._singular)
    PetscCallAbort(comm, MatNullSpaceCreate(comm, PETSC_TRUE, 0, nullptr, &ps.nullspace));

  // coarse levels: rediscretised and assembled, with the fine stencil's offsets
  ps.coarse.resize(levels - 1);
  for (const auto l : index_range(ps.coarse))
    ps.coarse[l] = PetscFD::createStencilMatrix(hierarchy.dms[l + 1], stencil);

  if (field._singular)
  {
    std::vector<Mat> all_levels{ps.fine};
    all_levels.insert(all_levels.end(), ps.coarse.begin(), ps.coarse.end());
    for (auto & A : all_levels)
    {
      PetscCallAbort(comm, MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
      PetscCallAbort(comm, MatSetOption(A, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
      PetscCallAbort(comm, MatSetNullSpace(A, ps.nullspace));
    }
  }
  else
  {
    // MAT_SPD of the coarse levels is set after every assembly (assembleCoarseLevels)
    PetscCallAbort(comm, MatSetOption(ps.fine, MAT_SPD, PETSC_TRUE));
    PetscCallAbort(comm, MatSetOption(ps.fine, MAT_SPD_ETERNAL, PETSC_TRUE));
  }

  // KSPCG + PCMG defaults, set before KSPSetFromOptions so that user options win
  PetscCallAbort(comm, KSPCreate(comm, &ps.ksp));
  PetscCallAbort(comm, KSPSetOptionsPrefix(ps.ksp, petscPrefix(field._name).c_str()));
  PetscCallAbort(comm, KSPSetOperators(ps.ksp, ps.fine, ps.fine));
  PetscCallAbort(comm, KSPSetType(ps.ksp, KSPCG));
  PetscCallAbort(comm, KSPSetNormType(ps.ksp, KSP_NORM_UNPRECONDITIONED));
  // no divergence test, as in TORCH_PCG: with the warm start x0 = u_old the residual is
  // compared to ||b|| ~ ||u_old|| / dt, so at large dt ||r0|| / ||b|| ~ dt ||B u_old|| / ||u_old||
  // already nears the default divtol and CG's nonmonotone residual crosses it
  PetscCallAbort(comm,
                 KSPSetTolerances(ps.ksp, _l_tol, PETSC_CURRENT, PETSC_UNLIMITED, _l_max_its));
  PetscCallAbort(comm, KSPSetInitialGuessNonzero(ps.ksp, PETSC_TRUE));

  PC pc;
  PetscCallAbort(comm, KSPGetPC(ps.ksp, &pc));
  PetscCallAbort(comm, PCSetType(pc, PCMG));
  PetscCallAbort(comm, PCMGSetLevels(pc, PetscInt(levels), nullptr));
  // PCMG numbers the levels coarsest first: PCMG level i is hierarchy level levels - 1 - i
  for (const auto i : make_range(1u, levels))
  {
    const auto h = levels - 1 - i;
    PetscCallAbort(comm, PCMGSetInterpolation(pc, PetscInt(i), hierarchy.interpolations[h]));
  }
  for (const auto i : make_range(levels - 1))
  {
    KSP smoother;
    PetscCallAbort(comm, PCMGGetSmoother(pc, PetscInt(i), &smoother));
    const auto B = ps.coarse[levels - 2 - i];
    PetscCallAbort(comm, KSPSetOperators(smoother, B, B));
  }
  // PCMG's default smoother is Chebyshev + SOR. The fine shell has no SOR, so it uses
  // Jacobi from its diagonal. On CUDA the coarse levels use Jacobi too, since PETSc's SOR
  // runs on the host. Chebyshev estimates its eigenvalue bounds with a few Krylov steps;
  // DAMPED_JACOBI (Richardson + Jacobi, weighted in assembleCoarseLevels) needs no
  // estimate. PCMG level 0 is the coarse LU when levels > 1.
  for (const auto i : make_range(levels > 1 ? 1u : 0u, levels))
    if (_damped_jacobi || i == levels - 1 || cuda)
    {
      KSP smoother;
      PC smoother_pc;
      PetscCallAbort(comm, PCMGGetSmoother(pc, PetscInt(i), &smoother));
      if (_damped_jacobi)
        PetscCallAbort(comm, KSPSetType(smoother, KSPRICHARDSON));
      PetscCallAbort(comm, KSPGetPC(smoother, &smoother_pc));
      PetscCallAbort(comm, PCSetType(smoother_pc, PCJACOBI));
    }
  if (field._singular && levels > 1)
  {
    // the coarse LU meets a zero pivot from the constant nullspace; shifting it changes
    // only the preconditioner
    KSP coarse_ksp;
    PC coarse_pc;
    PetscCallAbort(comm, PCMGGetCoarseSolve(pc, &coarse_ksp));
    PetscCallAbort(comm, KSPGetPC(coarse_ksp, &coarse_pc));
    PetscCallAbort(comm, PCFactorSetShiftType(coarse_pc, MAT_SHIFT_NONZERO));
  }

  assembleCoarseLevels(field);

  insertPetscOptions(petscPrefix(field._name), false);
  // PCMG would build new levels without the hierarchy above (and fail in PCSetUp)
  PetscBool user_levels;
  PetscCallAbort(comm,
                 PetscOptionsHasName(
                     nullptr, petscPrefix(field._name).c_str(), "-pc_mg_levels", &user_levels));
  if (user_levels)
    mooseError("-",
               petscPrefix(field._name),
               "pc_mg_levels is not supported: ImplicitFDSolver sets the number of multigrid "
               "levels from the grid and the stencil width.");
  PetscCallAbort(comm, KSPSetFromOptions(ps.ksp));

  // b and x get the solver's tensors placed before every solve
  if (cuda)
  {
#if MARLIN_PETSC_CUDA
    PetscCallAbort(comm, VecCreateMPICUDAWithArray(comm, 1, local, global, nullptr, &ps.b));
    PetscCallAbort(comm, VecCreateMPICUDAWithArray(comm, 1, local, global, nullptr, &ps.x));
#endif
  }
  else
  {
    PetscCallAbort(comm, VecCreateMPIWithArray(comm, 1, local, global, nullptr, &ps.b));
    PetscCallAbort(comm, VecCreateMPIWithArray(comm, 1, local, global, nullptr, &ps.x));
  }
}

void
ImplicitFDSolver::assembleCoarseLevels(Field & field)
{
  auto & ps = *field._petsc;
  const auto comm = ps.comm;
  const auto stencil = fieldStencil(field._fd_operators);
  // a per-cell field compares on the owned cells only, so every rank must reassemble when any
  // rank's stencil changed (the assembly is collective)
  int same = sameStencil(stencil, ps.assembled_stencil);
  PetscCallMPIAbort(comm, MPI_Allreduce(MPI_IN_PLACE, &same, 1, MPI_INT, MPI_MIN, comm));
  if (same)
    return;
  // PETSc's smoothers would divide by a zero diagonal
  checkPositiveDiagonal(field, jacobiDiagonal(field));
  const auto & grid_spacing = _domain.getGridSpacing();
  const auto levels = ps.coarse.size() + 1;
  // the coarse rows next to a boundary face fold the ghosts with the homogeneous
  // condition at the level's spacing (the boundary values enter the fine residual only)
  const auto faces = faceGhosts(field._face_bcs);
  PC pc;
  PetscCallAbort(comm, KSPGetPC(ps.ksp, &pc));
  // the smoother weights are set only for PCMG (not e.g. for -<buffer>_pc_type jacobi)
  PetscBool is_mg;
  PetscCallAbort(comm, PetscObjectTypeCompare((PetscObject)pc, PCMG, &is_mg));
  const bool set_weights = is_mg && _damped_jacobi;

  // a FiniteDifferenceDivGrad's coefficient on hierarchy level k > 0 is the mean of D over
  // the 2^dim children on level k - 1, and its coarse operator is rediscretised from it
  std::vector<std::vector<Vec>> coefficients(field._fd_operators.size());
  for (const auto i : index_range(field._fd_operators))
    if (const auto * div_grad =
            dynamic_cast<const FiniteDifferenceDivGrad *>(field._fd_operators[i]))
    {
      auto & D = coefficients[i];
      D.resize(levels);
      // the owned block of a Marlin tensor is this rank's part of the fine DMDA vector
      const auto fine_D =
          div_grad->ownedCoefficient().to(torch::kCPU, torch::kFloat64).contiguous();
      PetscCallAbort(comm, DMCreateGlobalVector(ps.hierarchy->dms[0], &D[0]));
      PetscScalar * array;
      PetscCallAbort(comm, VecGetArrayWrite(D[0], &array));
      std::copy_n(fine_D.data_ptr<double>(), fine_D.numel(), array);
      PetscCallAbort(comm, VecRestoreArrayWrite(D[0], &array));
      for (const auto k : make_range(std::size_t(1), levels))
      {
        PetscCallAbort(comm, DMCreateGlobalVector(ps.hierarchy->dms[k], &D[k]));
        PetscFD::restrictCellMean(ps.hierarchy->dms[k], ps.hierarchy->dms[k - 1], D[k - 1], D[k]);
      }
    }

  // hierarchy level k (0 = fine) has spacing 2^k h and is PCMG level levels - 1 - k
  for (const auto k : make_range(levels))
  {
    std::vector<Real> spacing(_dim);
    for (const auto d : make_range(_dim))
      spacing[d] = grid_spacing(d) * Real(1 << k);
    std::vector<FDStencilEntry> level_stencil;
    for (const auto i : index_range(field._fd_operators))
    {
      const auto * fd_op = field._fd_operators[i];
      // the coarse D with one halo layer: D_0 outside a domain face, as on the fine level
      const auto entries =
          k > 0 && !coefficients[i].empty()
              ? dynamic_cast<const FiniteDifferenceDivGrad *>(fd_op)->stencilFromCoefficient(
                    PetscFD::ownedWithHalo(ps.hierarchy->dms[k], coefficients[i][k]), spacing)
              : fd_op->stencilAt(spacing);
      level_stencil.insert(level_stencil.end(), entries.begin(), entries.end());
    }

    if (k > 0)
    {
      // PCMG restricts with P^T, whose column sums are 2^dim per level, so the
      // rediscretised operator is scaled by 2^(dim k) to match P^T A P on smooth fields.
      // This scales only the preconditioner.
      const PetscReal scale = std::pow(2.0, Real(_dim * k));
      auto & B = ps.coarse[k - 1];
      PetscFD::assembleStencilOperator(
          ps.hierarchy->dms[k], level_stencil, scale, faces, spacing, B);
      // assembly clears MAT_SPD
      if (!field._singular)
      {
        PetscCallAbort(comm, MatSetOption(B, MAT_SPD, PETSC_TRUE));
        PetscCallAbort(comm, MatSetOption(B, MAT_SPD_ETERNAL, PETSC_TRUE));
      }
      // without a DM, PCMG only sets up the fine smoother again on a new matrix; resetting
      // the operators makes the coarse smoothers redo theirs too (Chebyshev bounds)
      if (is_mg)
      {
        KSP smoother;
        PetscCallAbort(comm, PCMGGetSmoother(pc, PetscInt(levels - 1 - k), &smoother));
        PetscCallAbort(comm, KSPSetOperators(smoother, B, B));
      }
    }

    // damped Jacobi weight of the level smoother (PCMG level 0 is the coarse LU when
    // levels > 1). The spectrum of D^-1 A lies in [0, rho], rho the Gershgorin bound, the
    // maximum over rows including those next to a boundary face (the 2^(dim k) scaling
    // cancels); omega = 4 / (3 rho) damps the upper half [rho/2, rho] by at least 1/3 per
    // sweep and keeps |1 - omega lambda| < 1 on (0, rho].
    const auto pcmg_level = levels - 1 - k;
    if (!set_weights || (levels > 1 && pcmg_level == 0))
      continue;
    KSP smoother;
    PetscCallAbort(comm, PCMGGetSmoother(pc, PetscInt(pcmg_level), &smoother));
    // a user set -<buffer>_mg_levels_ksp_richardson_scale wins
    const char * prefix;
    PetscBool user_scale;
    PetscCallAbort(comm, KSPGetOptionsPrefix(smoother, &prefix));
    PetscCallAbort(comm,
                   PetscOptionsHasName(nullptr, prefix, "-ksp_richardson_scale", &user_scale));
    // KSPRichardsonSetScale is a no-op if the user changed the smoother type
    if (!user_scale)
      PetscCallAbort(comm,
                     KSPRichardsonSetScale(
                         smoother,
                         4.0 / (3.0 * PetscFD::jacobiSpectrumBound(
                                          ps.hierarchy->dms[k], level_stencil, faces, spacing))));
  }

  for (auto & D : coefficients)
    for (auto & v : D)
      PetscCallAbort(comm, VecDestroy(&v));

  // the shell's action changed too; bump its state and reset the KSP operators so
  // that PCMG sets up again (the fine smoother, the coarse factorisation)
  PetscCallAbort(comm, PetscObjectStateIncrease((PetscObject)ps.fine));
  PetscCallAbort(comm, KSPSetOperators(ps.ksp, ps.fine, ps.fine));
  ps.assembled_stencil = stencil;
}

ImplicitFDSolver::SolveResult
ImplicitFDSolver::solvePetsc(Field & field, torch::Tensor & b, torch::Tensor & x)
{
  if (!field._petsc)
    setupPetsc(field);
  else
    assembleCoarseLevels(field);
  auto & ps = *field._petsc;
  const auto comm = ps.comm;

  mooseAssert(b.is_contiguous() && x.is_contiguous(), "b and x must be contiguous");
  if (onCuda())
  {
#if MARLIN_PETSC_CUDA
    // b and x were written on torch's stream; KSPSolve runs on PETSc's
    cudaStream_t petsc_stream;
    PetscCallAbort(comm, PetscGetCurrentCUDAStream(&petsc_stream));
    const auto petsc = c10::cuda::getStreamFromExternal(petsc_stream, c10::cuda::current_device());
    at::cuda::CUDAEvent written;
    written.record(c10::cuda::getCurrentCUDAStream());
    written.block(petsc);
    PetscCallAbort(comm, VecCUDAPlaceArray(ps.b, b.data_ptr<PetscScalar>()));
    PetscCallAbort(comm, VecCUDAPlaceArray(ps.x, x.data_ptr<PetscScalar>()));
#endif
  }
  else
  {
    PetscCallAbort(comm, VecPlaceArray(ps.b, b.data_ptr<PetscScalar>()));
    PetscCallAbort(comm, VecPlaceArray(ps.x, x.data_ptr<PetscScalar>()));
  }

  ps.exception = nullptr;
  const auto ierr = KSPSolve(ps.ksp, ps.b, ps.x);

  if (onCuda())
  {
#if MARLIN_PETSC_CUDA
    PetscCallAbort(comm, VecCUDAResetArray(ps.b));
    PetscCallAbort(comm, VecCUDAResetArray(ps.x));
    // and torch reads x after PETSc has written it
    cudaStream_t petsc_stream;
    PetscCallAbort(comm, PetscGetCurrentCUDAStream(&petsc_stream));
    at::cuda::CUDAEvent solved;
    solved.record(c10::cuda::getStreamFromExternal(petsc_stream, c10::cuda::current_device()));
    solved.block(c10::cuda::getCurrentCUDAStream());
#endif
  }
  else
  {
    PetscCallAbort(comm, VecResetArray(ps.b));
    PetscCallAbort(comm, VecResetArray(ps.x));
  }

  // an exception from the operator kernels (caught in shellMult) is rethrown here, outside
  // PETSc's frames
  if (ps.exception)
    std::rethrow_exception(ps.exception);
  PetscCallAbort(comm, ierr);

  KSPConvergedReason reason;
  PetscInt iterations;
  PetscCallAbort(comm, KSPGetConvergedReason(ps.ksp, &reason));
  PetscCallAbort(comm, KSPGetIterationNumber(ps.ksp, &iterations));
  if (reason < 0)
    return {static_cast<unsigned int>(iterations),
            "The PETSC solve of field '" + field._name + "' failed after " +
                std::to_string(iterations) + " iterations: " + KSPConvergedReasons[reason] + "."};
  return {static_cast<unsigned int>(iterations), ""};
}

PetscErrorCode
ImplicitFDSolver::shellMult(Mat A, Vec x, Vec y)
{
  PetscSolve * ps;
  const PetscScalar * x_array;
  PetscScalar * y_array;
  PetscMemType x_memtype, y_memtype;
  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &ps));
  PetscCall(VecGetArrayReadAndMemType(x, &x_array, &x_memtype));
  PetscCall(VecGetArrayWriteAndMemType(y, &y_array, &y_memtype));
  PetscCheck(PetscMemTypeDevice(x_memtype) == PetscMemTypeDevice(y_memtype),
             PETSC_COMM_SELF,
             PETSC_ERR_ARG_INCOMP,
             "x and y must live on the same device");

  auto options = MooseTensor::floatTensorOptions().device(torch::kCPU);
#if MARLIN_PETSC_CUDA
  std::unique_ptr<PetscStreamGuard> guard;
  if (PetscMemTypeDevice(x_memtype))
  {
    options = options.device(torch::kCUDA, c10::cuda::current_device());
    guard = std::make_unique<PetscStreamGuard>();
  }
#endif
  // C++ exceptions must not unwind through PETSc's C frames: keep it for solvePetsc
  PetscErrorCode ierr = PETSC_SUCCESS;
  try
  {
    const auto shape = ps->field->_u_base.ownedView().sizes().vec();
    const auto in = torch::from_blob(const_cast<PetscScalar *>(x_array), shape, options);
    torch::from_blob(y_array, shape, options)
        .copy_(ps->solver->applyLinearOperator(*ps->field, in));
  }
  catch (...)
  {
    ps->exception = std::current_exception();
    ierr = PETSC_ERR_LIB;
  }

  PetscCall(VecRestoreArrayReadAndMemType(x, &x_array));
  PetscCall(VecRestoreArrayWriteAndMemType(y, &y_array));
  PetscCheck(
      ierr == PETSC_SUCCESS, PETSC_COMM_SELF, ierr, "The operator action threw an exception");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode
ImplicitFDSolver::shellDiagonal(Mat A, Vec d)
{
  PetscSolve * ps;
  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &ps));
  // with boundary faces the diagonal varies next to them: copy the owned tensor into d,
  // which lives on the host or (VECCUDA) on the device
  PetscScalar * array;
  PetscMemType memtype;
  PetscCall(VecGetArrayWriteAndMemType(d, &array, &memtype));
  auto options = MooseTensor::floatTensorOptions().device(torch::kCPU);
#if MARLIN_PETSC_CUDA
  std::unique_ptr<PetscStreamGuard> guard;
  if (PetscMemTypeDevice(memtype))
  {
    options = options.device(torch::kCUDA, c10::cuda::current_device());
    guard = std::make_unique<PetscStreamGuard>();
  }
#endif
  // C++ exceptions must not unwind through PETSc's C frames: keep it for solvePetsc
  PetscErrorCode ierr = PETSC_SUCCESS;
  try
  {
    const auto diag = ps->solver->jacobiDiagonal(*ps->field);
    torch::from_blob(array, diag.sizes(), options).copy_(diag);
  }
  catch (...)
  {
    ps->exception = std::current_exception();
    ierr = PETSC_ERR_LIB;
  }
  PetscCall(VecRestoreArrayWriteAndMemType(d, &array));
  PetscCheck(
      ierr == PETSC_SUCCESS, PETSC_COMM_SELF, ierr, "The Jacobi diagonal threw an exception");
  PetscFunctionReturn(PETSC_SUCCESS);
}

void
ImplicitFDSolver::nonlinearSubstep()
{
  // the explicit computes run at the time level of the unknown, so the converged step is
  // backward Euler A U = u^n/dt + f(U, t + dt) (D15); the boundary functions stay at t + dt
  // through the shifted time, and the buffer-valued boundary data follow every iterate (D16)
  _sub_time += _sub_dt;
  _tensor_problem.beginFixedBoundaryData(0.0);

  // a singular field's solution is the zero-mean one, as in the lagged scheme; the corrections
  // have zero mean, so the initial iterate is projected
  std::vector<torch::Tensor> U;
  for (auto & field : _fields)
  {
    U.push_back(field._u_base.ownedView().clone(torch::MemoryFormat::Contiguous));
    if (field._singular)
      U.back() -=
          _domain.allreduceSum(U.back().sum()) / static_cast<Real>(_domain.getNumberOfCells());
  }

  _linear_iterations = 0;
  SolveResult result;
  try
  {
    result = _nonlinear_solver == NonlinearSolver::PETSC ? solveNonlinearPetsc(U)
                                                         : solveNonlinearTorch(U);
  }
  catch (...)
  {
    _tensor_problem.endFixedBoundaryData();
    _sub_time -= _sub_dt;
    throw;
  }
  // the time stepper's signal is the number of nonlinear iterations (D14)
  _iterations = result.iterations;
  _console << "Nonlinear solve: " << result.iterations << " iterations, " << _linear_iterations
           << " linear iterations." << std::endl;

  if (!result.failure.empty())
    failSubstep(result.failure);
  else
    // the fields, the explicit terms and the operator output buffers at the converged U
    nonlinearRhs(U);

  _tensor_problem.endFixedBoundaryData();
  _sub_time -= _sub_dt;
}

std::vector<torch::Tensor>
ImplicitFDSolver::nonlinearRhs(const std::vector<torch::Tensor> & U)
{
  // every field holds its iterate before any explicit compute runs (Jacobi, D11), so b(U) does
  // not depend on the field order. No ghost exchange here: every compute reading a field's ghosts
  // exchanges them itself (runComputeWithGhosts)
  for (const auto i : index_range(_fields))
    applyOperator(_fields[i], U[i]);
  for (auto * cmp : _explicit_computes)
    cmp->realSpaceComputeBuffer();
  forwardBuffers();
  // the buffer-valued boundary data from the explicit terms at U
  _tensor_problem.beginFixedBoundaryData(0.0);

  std::vector<torch::Tensor> b;
  for (auto & field : _fields)
    b.push_back(fieldRhs(field));

  // fieldRhs applied the operators to 0 (for the affine part): give their outputs A_i U again
  for (const auto i : index_range(_fields))
    if (hasBoundaryFaces(_fields[i]))
      applyOperator(_fields[i], U[i]);
  return b;
}

ImplicitFDSolver::SolveResult
ImplicitFDSolver::applyFieldSolves(const std::vector<torch::Tensor> & r,
                                   std::vector<torch::Tensor> & y)
{
  SolveResult total;
  for (const auto i : index_range(_fields))
  {
    auto & field = _fields[i];
    // the correction solves the linear part from x0 = 0, with the field's mean handling
    auto b = r[i].clone(torch::MemoryFormat::Contiguous);
    y[i] = torch::zeros_like(b, torch::MemoryFormat::Contiguous);
    if (field._singular)
      // the mean of a singular field's residual is the round-off fieldRhs left
      b -= _domain.allreduceSum(b.sum()) / static_cast<Real>(_domain.getNumberOfCells());
    const auto result = solveField(field, b, y[i]);
    total.iterations += result.iterations;
    if (!result.failure.empty())
    {
      total.failure = result.failure;
      return total;
    }
  }
  return total;
}

ImplicitFDSolver::SolveResult
ImplicitFDSolver::solveNonlinearTorch(std::vector<torch::Tensor> & U)
{
  // defect correction U <- U - P^-1 F(U), F(U) = A U - b(U), with P the field solves (as
  // SNESSetPicard with the full step), and the tests of SNESConvergedDefault on ||F||_2 (D12)
  std::vector<torch::Tensor> F(_fields.size()), dU(_fields.size());
  Real F0_norm = 0.0;
  for (unsigned int it = 0;; ++it)
  {
    const auto b = nonlinearRhs(U);
    Real F_norm_sq = 0.0;
    for (const auto i : index_range(_fields))
    {
      F[i] = applyLinearOperator(_fields[i], U[i]) - b[i];
      F_norm_sq += _domain.allreduceSum((F[i] * F[i]).sum()).cpu().item<double>();
    }
    const Real F_norm = std::sqrt(F_norm_sq);
    if (it == 0)
      F0_norm = F_norm;

    if (!std::isfinite(F_norm))
      return {it, "The nonlinear solve diverged: the residual norm is not finite."};
    if (F_norm < _nl_abs_tol || (it > 0 && F_norm <= _nl_rel_tol * F0_norm))
      return {it, ""};
    if (it == _nl_max_its)
    {
      std::ostringstream failure;
      failure << "The nonlinear solve did not converge in " << _nl_max_its
              << " iterations (relative residual " << F_norm / F0_norm
              << ", nl_rel_tol = " << _nl_rel_tol << ").";
      return {it, failure.str()};
    }

    const auto result = applyFieldSolves(F, dU);
    _linear_iterations += result.iterations;
    if (!result.failure.empty())
      return {it, result.failure};
    for (const auto i : index_range(_fields))
      U[i] -= dU[i];
  }
}

ImplicitFDSolver::NonlinearSolve::~NonlinearSolve()
{
  PetscCallAbort(PETSC_COMM_WORLD, SNESDestroy(&snes));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&A));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&x));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&r));
}

namespace
{
/// torch's current stream and PETSc's: work on one waits for the other (CUDA only)
void
syncStreams(bool petsc_waits)
{
#if MARLIN_PETSC_CUDA
  if (!onCuda())
    return;
  cudaStream_t petsc_stream;
  PetscCallAbort(PETSC_COMM_SELF, PetscGetCurrentCUDAStream(&petsc_stream));
  const auto petsc = c10::cuda::getStreamFromExternal(petsc_stream, c10::cuda::current_device());
  const auto torch_stream = c10::cuda::getCurrentCUDAStream();
  at::cuda::CUDAEvent done;
  done.record(petsc_waits ? torch_stream : petsc);
  done.block(petsc_waits ? petsc : torch_stream);
#else
  libmesh_ignore(petsc_waits);
#endif
}

/// copy a field tensor into a coupled view: move to PETSc's device, then cast (MPS holds no
/// double)
void
copyToCoupled(const torch::Tensor & view, const torch::Tensor & field)
{
  torch::Tensor out = view;
  out.copy_(field.to(view.device()));
}
}

ImplicitFDSolver::SolveResult
ImplicitFDSolver::solveNonlinearPetsc(std::vector<torch::Tensor> & U)
{
  if (!_nonlinear)
    setupNonlinearPetsc();
  auto & nl = *_nonlinear;
  const auto comm = PetscObjectComm((PetscObject)nl.snes);

  // the coupled iterate from the fields' owned blocks
  {
    PetscScalar * array;
    PetscMemType memtype;
    PetscCallAbort(comm, VecGetArrayWriteAndMemType(nl.x, &array, &memtype));
    const auto views = coupledViews(array, memtype);
    for (const auto i : index_range(_fields))
      copyToCoupled(views[i], U[i]);
    PetscCallAbort(comm, VecRestoreArrayWriteAndMemType(nl.x, &array));
  }

  syncStreams(true);
  nl.exception = nullptr;
  nl.failure.clear();
  const auto ierr = SNESSolve(nl.snes, nullptr, nl.x);
  syncStreams(false);
  // an exception from a callback is rethrown here, outside PETSc's frames
  if (nl.exception)
    std::rethrow_exception(nl.exception);
  PetscCallAbort(comm, ierr);

  {
    const PetscScalar * array;
    PetscMemType memtype;
    PetscCallAbort(comm, VecGetArrayReadAndMemType(nl.x, &array, &memtype));
    const auto tensors = toFieldTensors(coupledViews(const_cast<PetscScalar *>(array), memtype));
    for (const auto i : index_range(_fields))
      U[i].copy_(tensors[i]);
    PetscCallAbort(comm, VecRestoreArrayReadAndMemType(nl.x, &array));
  }

  SNESConvergedReason reason;
  PetscInt iterations;
  PetscCallAbort(comm, SNESGetConvergedReason(nl.snes, &reason));
  PetscCallAbort(comm, SNESGetIterationNumber(nl.snes, &iterations));
  if (reason < 0)
  {
    std::string failure = "The nonlinear solve failed after " + std::to_string(iterations) +
                          " iterations: " + SNESConvergedReasons[reason] + ".";
    if (!nl.failure.empty())
      failure += " " + nl.failure;
    return {static_cast<unsigned int>(iterations), failure};
  }
  return {static_cast<unsigned int>(iterations), ""};
}

void
ImplicitFDSolver::setupNonlinearPetsc()
{
  _nonlinear = std::make_unique<NonlinearSolve>();
  auto & nl = *_nonlinear;
  const auto comm = _domain.getMPIComm();

  // a coupled vector holds the owned cells of every field, in field order
  PetscInt local = 0;
  _coupled_offsets.clear();
  for (const auto & field : _fields)
  {
    _coupled_offsets.push_back(local);
    local += field._u_base.ownedView().numel();
  }
  const PetscInt global = PetscInt(_fields.size()) * _domain.getNumberOfCells();

  // the coupled vectors on the torch device where PETSc can hold them (CUDA with a CUDA PETSc),
  // on the host otherwise, where every iterate passes through a copy
  const bool cuda = onCuda() && MARLIN_PETSC_CUDA;
  if (cuda)
  {
#if MARLIN_PETSC_CUDA
    PetscCallAbort(comm, VecCreateMPICUDA(comm, local, global, &nl.x));
#endif
  }
  else
    PetscCallAbort(comm, VecCreateMPI(comm, local, global, &nl.x));
  PetscCallAbort(comm, VecDuplicate(nl.x, &nl.r));
  const auto device = MooseTensor::floatTensorOptions().device();
  if (!cuda && !device.is_cpu())
    mooseWarning("PETSc's nonlinear solve keeps its vectors on the host while the field solves "
                 "run on '",
                 device.str(),
                 "'",
                 device.is_cuda() ? " (PETSc is not built with CUDA)" : "",
                 ": every residual, operator product and preconditioner application copies "
                 "the iterate between them. This works but is slow; nonlinear_solver = TORCH "
                 "keeps the iteration on the device.");

  // A, block-diagonal over the fields: the Picard operator of SNESSetPicard
  PetscCallAbort(comm, MatCreateShell(comm, local, local, global, global, this, &nl.A));
  PetscCallAbort(comm, MatShellSetOperation(nl.A, MATOP_MULT, (void (*)(void))coupledMult));
#if MARLIN_PETSC_CUDA
  if (cuda)
    PetscCallAbort(comm, MatShellSetVecType(nl.A, VECCUDA));
#endif

  // F(U) = A U - b(U) with A as the Jacobian (right preconditioning by the field solves)
  PetscCallAbort(comm, SNESCreate(comm, &nl.snes));
  PetscCallAbort(comm, SNESSetOptionsPrefix(nl.snes, "nonlinear_"));
  PetscCallAbort(
      comm,
      SNESSetPicard(
          nl.snes, nl.r, nonlinearRhsCallback, nl.A, nl.A, nonlinearJacobianCallback, this));
  // defaults, set before SNESSetFromOptions so that user options win: the full Picard defect
  // correction (D13); the tests on ||F||_2 alone, no step length test (D12), follow below
  PetscCallAbort(comm, SNESSetType(nl.snes, SNESNEWTONLS));
  SNESLineSearch linesearch;
  PetscCallAbort(comm, SNESGetLineSearch(nl.snes, &linesearch));
  PetscCallAbort(comm, SNESLineSearchSetType(linesearch, SNESLINESEARCHNONE));
  setFieldSolvePreconditioner(nl.snes);

  insertPetscOptions("nonlinear_", true);
  PetscCallAbort(comm, SNESSetFromOptions(nl.snes));

  // the tolerances after the options, since a type set there (e.g. anderson) resets them to its
  // own defaults (10000 iterations); those given as options win. No divergence test, as in the
  // torch loop: a growing residual fails at nl_max_its
  const auto given = [comm](const char * name)
  {
    PetscBool set;
    PetscCallAbort(comm, PetscOptionsHasName(nullptr, "nonlinear_", name, &set));
    return bool(set);
  };
  const auto current = PetscReal(PETSC_CURRENT);
  PetscCallAbort(comm,
                 SNESSetTolerances(nl.snes,
                                   given("-snes_atol") ? current : _nl_abs_tol,
                                   given("-snes_rtol") ? current : _nl_rel_tol,
                                   given("-snes_stol") ? current : 0.0,
                                   given("-snes_max_it") ? PETSC_CURRENT : PetscInt(_nl_max_its),
                                   PETSC_CURRENT));
  if (!given("-snes_divergence_tolerance"))
    PetscCallAbort(comm, SNESSetDivergenceTolerance(nl.snes, PETSC_UNLIMITED));

  // a nonlinear preconditioner (-nonlinear_npc_snes_type, e.g. under anderson) uses the field
  // solves and the full step too, unless its options say otherwise (read again in SNESSetUp)
  PetscBool has_npc;
  PetscCallAbort(comm, SNESHasNPC(nl.snes, &has_npc));
  if (has_npc)
  {
    SNES npc;
    PetscCallAbort(comm, SNESGetNPC(nl.snes, &npc));
    PetscCallAbort(comm, SNESGetLineSearch(npc, &linesearch));
    PetscCallAbort(comm, SNESLineSearchSetType(linesearch, SNESLINESEARCHNONE));
    setFieldSolvePreconditioner(npc);
  }
}

void
ImplicitFDSolver::setFieldSolvePreconditioner(SNES snes)
{
  const auto comm = PetscObjectComm((PetscObject)snes);
  KSP ksp;
  PC pc;
  PetscCallAbort(comm, SNESGetKSP(snes, &ksp));
  // one application of the field solves (each converged to l_tol) per correction
  PetscCallAbort(comm, KSPSetType(ksp, KSPPREONLY));
  PetscCallAbort(comm, KSPGetPC(ksp, &pc));
  PetscCallAbort(comm, PCSetType(pc, PCSHELL));
  PetscCallAbort(comm, PCShellSetContext(pc, this));
  PetscCallAbort(comm, PCShellSetApply(pc, fieldSolvePCApply));
  PetscCallAbort(comm, PCShellSetName(pc, "ImplicitFDSolver field solves"));
}

std::vector<torch::Tensor>
ImplicitFDSolver::coupledViews(PetscScalar * array, PetscMemType memtype) const
{
  // PETSc's scalar type, which is not torch's with the field solves in single precision
  auto options = torch::TensorOptions()
                     .dtype(c10::CppTypeToScalarType<PetscScalar>::value)
                     .device(torch::kCPU);
#if MARLIN_PETSC_CUDA
  if (PetscMemTypeDevice(memtype))
    options = options.device(torch::kCUDA, c10::cuda::current_device());
#else
  libmesh_ignore(memtype);
#endif
  std::vector<torch::Tensor> views;
  for (const auto i : index_range(_fields))
    views.push_back(torch::from_blob(
        array + _coupled_offsets[i], _fields[i]._u_base.ownedView().sizes().vec(), options));
  return views;
}

std::vector<torch::Tensor>
ImplicitFDSolver::toFieldTensors(const std::vector<torch::Tensor> & views) const
{
  // cast on PETSc's side, then move: MPS holds no double; no copy when device and dtype match
  const auto options = MooseTensor::floatTensorOptions();
  std::vector<torch::Tensor> tensors;
  for (const auto & view : views)
    tensors.push_back(view.to(options.dtype()).to(options.device()));
  return tensors;
}

namespace
{
/// run f(x_array, x_memtype, y_array, y_memtype) on the arrays of x (read) and y (written), with
/// torch on PETSc's stream; a C++ exception must not unwind through PETSc's frames, so it is
/// kept in exception
template <typename F>
PetscErrorCode
withCoupledArrays(Vec x, Vec y, std::exception_ptr & exception, F && f)
{
  const PetscScalar * x_array;
  PetscScalar * y_array;
  PetscMemType x_memtype, y_memtype;
  PetscFunctionBeginUser;
  PetscCall(VecGetArrayReadAndMemType(x, &x_array, &x_memtype));
  PetscCall(VecGetArrayWriteAndMemType(y, &y_array, &y_memtype));
#if MARLIN_PETSC_CUDA
  std::unique_ptr<PetscStreamGuard> guard;
  if (PetscMemTypeDevice(x_memtype))
    guard = std::make_unique<PetscStreamGuard>();
#endif
  PetscErrorCode ierr = PETSC_SUCCESS;
  try
  {
    f(const_cast<PetscScalar *>(x_array), x_memtype, y_array, y_memtype);
  }
  catch (...)
  {
    exception = std::current_exception();
    ierr = PETSC_ERR_LIB;
  }
  PetscCall(VecRestoreArrayReadAndMemType(x, &x_array));
  PetscCall(VecRestoreArrayWriteAndMemType(y, &y_array));
  PetscCheck(ierr == PETSC_SUCCESS, PETSC_COMM_SELF, ierr, "A callback threw an exception");
  PetscFunctionReturn(PETSC_SUCCESS);
}
}

PetscErrorCode
ImplicitFDSolver::nonlinearRhsCallback(SNES, Vec x, Vec b, void * ctx)
{
  auto * solver = static_cast<ImplicitFDSolver *>(ctx);
  return withCoupledArrays(
      x,
      b,
      solver->_nonlinear->exception,
      [solver](PetscScalar * x_array, PetscMemType x_mt, PetscScalar * b_array, PetscMemType b_mt)
      {
        const auto rhs =
            solver->nonlinearRhs(solver->toFieldTensors(solver->coupledViews(x_array, x_mt)));
        const auto out = solver->coupledViews(b_array, b_mt);
        for (const auto i : index_range(out))
          copyToCoupled(out[i], rhs[i]);
      });
}

PetscErrorCode
ImplicitFDSolver::nonlinearJacobianCallback(SNES, Vec, Mat, Mat, void *)
{
  // nothing to assemble: the shell applies the operators as they stand (their coefficients are
  // explicit terms of the step)
  return PETSC_SUCCESS;
}

PetscErrorCode
ImplicitFDSolver::coupledMult(Mat A, Vec x, Vec y)
{
  ImplicitFDSolver * solver;
  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &solver));
  PetscCall(withCoupledArrays(
      x,
      y,
      solver->_nonlinear->exception,
      [solver](PetscScalar * x_array, PetscMemType x_mt, PetscScalar * y_array, PetscMemType y_mt)
      {
        const auto in = solver->toFieldTensors(solver->coupledViews(x_array, x_mt));
        const auto out = solver->coupledViews(y_array, y_mt);
        for (const auto i : index_range(out))
          copyToCoupled(out[i], solver->applyLinearOperator(solver->_fields[i], in[i]));
      }));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode
ImplicitFDSolver::fieldSolvePCApply(PC pc, Vec r, Vec y)
{
  ImplicitFDSolver * solver;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &solver));
  SolveResult result;
  PetscCall(withCoupledArrays(
      r,
      y,
      solver->_nonlinear->exception,
      [solver,
       &result](PetscScalar * r_array, PetscMemType r_mt, PetscScalar * y_array, PetscMemType y_mt)
      {
        const auto rhs = solver->toFieldTensors(solver->coupledViews(r_array, r_mt));
        std::vector<torch::Tensor> correction(rhs.size());
        result = solver->applyFieldSolves(rhs, correction);
        const auto out = solver->coupledViews(y_array, y_mt);
        for (const auto i : index_range(out))
          if (correction[i].defined())
            copyToCoupled(out[i], correction[i]);
      }));
  solver->_linear_iterations += result.iterations;
  // a failed field solve fails the preconditioner, and so the (preonly) KSP and the SNES
  if (!result.failure.empty())
    solver->_nonlinear->failure = result.failure;
  PetscCall(PCSetFailedReason(pc, result.failure.empty() ? PC_NOERROR : PC_SUBPC_ERROR));
  PetscFunctionReturn(PETSC_SUCCESS);
}
