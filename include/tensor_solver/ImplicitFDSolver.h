/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/
#pragma once

#include "TensorSolver.h"
#include "IterativeTensorSolverInterface.h"
#include "LinearFDOperator.h"
#include "PetscFDHierarchy.h"

#include <petscksp.h>
#include <petscsnes.h>

#include <exception>
#include <memory>

class TensorBufferBase;
class TensorRobinBC;

/**
 * Implicit solver for fields whose linear terms are LinearFDOperator computes.
 * Each field u solves sum_i A_i u = sum_i r_i + f, where A_i is the action of
 * the i-th operator as written and r_i its right-hand side contribution (e.g.
 * u_old/dt for TimeDerivativeOperator). The fine-level operator action reuses
 * each operator's own computeBuffer(), so it is exactly the explicit kernel.
 */
class ImplicitFDSolver : public TensorSolver, public IterativeTensorSolverInterface
{
public:
  static InputParameters validParams();

  ImplicitFDSolver(const InputParameters & parameters);

protected:
  struct Field;

  /// PETSc objects of one field's KSPCG + PCMG solve
  struct PetscSolve
  {
    PetscSolve() = default;
    PetscSolve(const PetscSolve &) = delete;
    PetscSolve & operator=(const PetscSolve &) = delete;
    ~PetscSolve();

    /// communicator with the ranks in DMDA order
    MPI_Comm comm = MPI_COMM_NULL;
    /// DMDA levels and interpolations (fine first)
    std::unique_ptr<PetscFD::Hierarchy> hierarchy;
    /// fine level shell operator and assembled coarse level operators (coarse[l - 1] is level l)
    Mat fine = nullptr;
    std::vector<Mat> coarse;
    /// constants, the nullspace of a singular field's operator on every level
    MatNullSpace nullspace = nullptr;
    KSP ksp = nullptr;
    /// right-hand side and solution, wrapping the solver's tensors
    Vec b = nullptr, x = nullptr;
    /// fine stencil the coarse levels were last assembled from
    std::vector<FDStencilEntry> assembled_stencil;
    /// MatMult context of the shell
    ImplicitFDSolver * solver = nullptr;
    Field * field = nullptr;
    /// exception thrown by the operator action inside KSPSolve (rethrown after it)
    std::exception_ptr exception;
  };

  struct Field
  {
    /// name and buffer of the solved field (doubles as Krylov scratch during the solve)
    std::string _name;
    torch::Tensor & _u;
    TensorBufferBase & _u_base;
    /// the linear operators acting on u, and their output buffers
    std::vector<TensorOperatorBase *> _operators;
    std::vector<const LinearFDOperator *> _fd_operators;
    std::vector<TensorBufferBase *> _operator_outputs;
    /// explicit source term f (null if none)
    TensorBufferBase * _source;
    /// sum_i A_i has the constants as its nullspace
    bool _singular;
    /// largest sum|b| of the singular field's solves so far, the scale of its round-off mean
    Real _max_b_abs_sum = 0.0;
    /// PETSc backend state (set up on the first solve)
    std::unique_ptr<PetscSolve> _petsc;
    /// boundary condition of u on each face (face 2d + upper; null on periodic faces)
    std::vector<const TensorRobinBC *> _face_bcs;
    /// sum_i A_i 0, the boundary data's part of the affine operator action (undefined
    /// without boundary faces); the linear operator sum_i A_i x applies them with zero face data
    torch::Tensor _affine;
    /// u^n and sum_i r_i (the time derivatives' u^n/dt) of the current substep
    torch::Tensor _u_old;
    torch::Tensor _rhs_old;
  };

  virtual void substep() override;

  /// iterations of a linear solve, and the reason it failed (empty if it converged)
  struct SolveResult
  {
    unsigned int iterations = 0;
    std::string failure;
  };

  /// collect the computes of the root compute other than the field operators
  void findExplicitComputes();

  /// the right-hand side of the linear part, sum_i r_i + f - g, from the explicit terms in the
  /// buffers (sets _affine = g; a singular field's round-off mean is removed)
  torch::Tensor fieldRhs(Field & field);

  /// solve the linear part of sum_i A_i x = b with the selected backend, warm started from x,
  /// with the mean of x fixed in the constant mode
  SolveResult solveField(Field & field, torch::Tensor & b, torch::Tensor & x);

  /// a failed solve: restore u^n of every field for the time stepper's retry with a smaller dt
  /// (or error without TensorSolveIterationAdaptiveDT)
  void failSubstep(const std::string & failure);

  /// y = sum_i A_i x on the owned cells (x is copied into u's owned cells first), with the
  /// boundary conditions' ghost values, so affine on a domain with boundary faces
  torch::Tensor applyOperator(Field & field, const torch::Tensor & x);

  /// the linear part of applyOperator: sum_i A_i x - sum_i A_i 0
  torch::Tensor applyLinearOperator(Field & field, const torch::Tensor & x);

  /// the Jacobi diagonal on the owned cells: the diagonal of the folded operator, which next to
  /// a boundary face includes the ghosts' dependence on the cell itself
  torch::Tensor jacobiDiagonal(const Field & field) const;
  /// error unless the folded diagonal is positive in every cell
  void checkPositiveDiagonal(const Field & field, const torch::Tensor & diag) const;

  /// lambda if the linear operator maps the constants to lambda > 0 times the constants (up to
  /// round-off), else 0
  Real constantEigenvalue(Field & field);

  /// solve sum_i A_i x = b (with ||b||_2 = b_norm) by torch preconditioned CG, warm started from x
  SolveResult solveTorchPCG(Field & field, const torch::Tensor & b, Real b_norm, torch::Tensor & x);

  /// solve sum_i A_i x = b by PETSc KSPCG + PCMG, warm started from x (both contiguous)
  SolveResult solvePetsc(Field & field, torch::Tensor & b, torch::Tensor & x);

  /// create the hierarchy, the level operators and the KSP of a field
  void setupPetsc(Field & field);

  /// (re)assemble the coarse levels and set the smoother weights if the stencil changed
  void assembleCoarseLevels(Field & field);

  /// put the block's PETSc options of the field solves (prefix <buffer>_) or of the nonlinear solve
  /// (prefix nonlinear_) and the command line into the global database
  void insertPetscOptions(const std::string & prefix, bool nonlinear);

  /// MATOP_MULT and MATOP_GET_DIAGONAL of the fine level shell
  static PetscErrorCode shellMult(Mat A, Vec x, Vec y);
  static PetscErrorCode shellDiagonal(Mat A, Vec d);

  /// PETSc objects of the coupled nonlinear solve over all fields
  struct NonlinearSolve
  {
    NonlinearSolve() = default;
    NonlinearSolve(const NonlinearSolve &) = delete;
    NonlinearSolve & operator=(const NonlinearSolve &) = delete;
    ~NonlinearSolve();

    SNES snes = nullptr;
    /// block-diagonal shell of the fields' linear operators
    Mat A = nullptr;
    /// coupled iterate and residual: the owned cells of every field, in field order
    Vec x = nullptr, r = nullptr;
    /// exception thrown inside a callback (rethrown after SNESSolve)
    std::exception_ptr exception;
    /// reason of the last failed field solve inside the preconditioner
    std::string failure;
  };

  /// iterate the coupled step A U = b(U) to convergence (u^n and sum_i r_i are set)
  void nonlinearSubstep();

  /// b(U) of every field: the explicit computes and the buffer-valued boundary data
  /// re-evaluated at U (Jacobi), then fieldRhs
  std::vector<torch::Tensor> nonlinearRhs(const std::vector<torch::Tensor> & U);

  /// the block-diagonal preconditioner: y_i solves the linear part of field i with r_i, from 0
  SolveResult applyFieldSolves(const std::vector<torch::Tensor> & r,
                               std::vector<torch::Tensor> & y);

  /// defect correction U <- U - P^-1 F(U), with P the field solves (TORCH_PCG backend)
  SolveResult solveNonlinearTorch(std::vector<torch::Tensor> & U);

  /// SNES with the field solves as the preconditioner of A (PETSC backend)
  SolveResult solveNonlinearPetsc(std::vector<torch::Tensor> & U);
  /// create the coupled vectors, the shell A and the SNES (prefix nonlinear_)
  void setupNonlinearPetsc();
  /// preonly KSP with the field solves as a shell preconditioner
  void setFieldSolvePreconditioner(SNES snes);

  /// the fields' owned blocks of a coupled vector's local array
  std::vector<torch::Tensor> coupledViews(PetscScalar * array, PetscMemType memtype) const;

  /// SNESSetPicard callbacks, the coupled shell's MATOP_MULT and the shell preconditioner
  static PetscErrorCode nonlinearRhsCallback(SNES snes, Vec x, Vec b, void * ctx);
  static PetscErrorCode nonlinearJacobianCallback(SNES snes, Vec x, Mat A, Mat P, void * ctx);
  static PetscErrorCode coupledMult(Mat A, Vec x, Vec y);
  static PetscErrorCode fieldSolvePCApply(PC pc, Vec r, Vec y);

  enum class LinearSolver
  {
    PETSC,
    TORCH_PCG
  };

  /// selected linear solver backend
  LinearSolver _linear_solver;

  /// relative tolerance on the unpreconditioned residual 2-norm
  const Real _l_tol;

  /// maximum number of linear iterations
  const unsigned int _l_max_its;

  /// smooth the PCMG levels by Gershgorin-weighted damped Jacobi instead of Chebyshev
  const bool _damped_jacobi;

  /// iterate the coupled nonlinear step instead of lagging the explicit terms
  const bool _nonlinear_solve;
  /// relative and absolute tolerances on ||A U - b(U)||_2 and the iteration limit
  const Real _nl_rel_tol;
  const Real _nl_abs_tol;
  const unsigned int _nl_max_its;
  /// linear iterations of the field solves in the current nonlinear solve
  unsigned int _linear_iterations = 0;
  /// offset of each field's owned block in a coupled vector
  std::vector<PetscInt> _coupled_offsets;
  std::unique_ptr<NonlinearSolve> _nonlinear;

  std::vector<Field> _fields;

  /// computes of the root compute that update the explicit terms (not the field operators)
  std::vector<TensorOperatorBase *> _explicit_computes;
  bool _explicit_computes_found = false;
};
