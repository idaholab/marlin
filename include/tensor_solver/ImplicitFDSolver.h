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

#include <exception>
#include <memory>

class TensorBufferBase;

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
    /// PETSc backend state (set up on the first solve)
    std::unique_ptr<PetscSolve> _petsc;
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

  /// y = sum_i A_i x on the owned cells (x is copied into u's owned cells first)
  torch::Tensor applyOperator(Field & field, const torch::Tensor & x);

  /// sum of the operators' centre stencil coefficients (the Jacobi diagonal)
  Real diagonal(const Field & field) const;

  /// solve sum_i A_i x = b (with ||b||_2 = b_norm) by torch preconditioned CG, warm started from x
  SolveResult solveTorchPCG(Field & field, const torch::Tensor & b, Real b_norm, torch::Tensor & x);

  /// solve sum_i A_i x = b by PETSc KSPCG + PCMG, warm started from x (both contiguous)
  SolveResult solvePetsc(Field & field, torch::Tensor & b, torch::Tensor & x);

  /// create the hierarchy, the level operators and the KSP of a field
  void setupPetsc(Field & field);

  /// (re)assemble the coarse levels and set the smoother weights if the stencil changed
  void assembleCoarseLevels(Field & field);

  /// put the block's PETSc options (prefix <buffer>_) and the command line into the global database
  void insertPetscOptions(const Field & field);

  /// MATOP_MULT and MATOP_GET_DIAGONAL of the fine level shell
  static PetscErrorCode shellMult(Mat A, Vec x, Vec y);
  static PetscErrorCode shellDiagonal(Mat A, Vec d);

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

  std::vector<Field> _fields;

  /// computes of the root compute that update the explicit terms (not the field operators)
  std::vector<TensorOperatorBase *> _explicit_computes;
  bool _explicit_computes_found = false;
};
