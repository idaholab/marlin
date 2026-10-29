/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorBoundaryCondition.h"
#include "FunctionInterface.h"
#include "RobinGhost.h"

/**
 * General Robin boundary condition: a*u + b*du/dn = c on the given face,
 * where n is the outward normal. The ghost fill (fillGhostCells) and the implicit
 * solver's matrix entries share the ghost relation in RobinGhost.h, so the two cannot
 * drift apart.
 * c may vary along the face (c_function or c_buffer); a and b are constants.
 */
class TensorRobinBC : public TensorBoundaryCondition, public FunctionInterface
{
public:
  static InputParameters validParams();

  TensorRobinBC(const InputParameters & parameters);

  virtual void fillGhostCells(torch::Tensor & buffer,
                              unsigned int ghost_layers,
                              int64_t halo,
                              int64_t owned_extent,
                              Real dx) const override;

  /// exchange the halo of c_buffer along the face
  virtual void prepareFaceData() override;

  virtual void checkGhostLayers(unsigned int ghost_layers, Real dx) const override;

  /// the first ghost layer l < ghost_layers whose denominator a/2 + b/((2l + 1) dx) is zero,
  /// or ghost_layers if there is none
  static unsigned int zeroDenominatorLayer(Real a, Real b, Real dx, unsigned int ghost_layers);

  /// Object-independent core of fillGhostCells(), used by the instance method and the unit test.
  /// c is a scalar tensor or has the shape of a ghost slice (size 1 along face_dim). Ghost layer
  /// l uses the relation of RobinGhost.h with the spacing (2l + 1) dx.
  static void fillGhostCellsStatic(torch::Tensor & buffer,
                                   bool is_upper_face,
                                   int64_t face_dim,
                                   unsigned int ghost_layers,
                                   int64_t halo,
                                   int64_t owned_extent,
                                   Real a,
                                   Real b,
                                   const torch::Tensor & c,
                                   Real dx);

  /// the constant Robin coefficients a and b
  Real a() const { return _a; }
  Real b() const { return _b; }

protected:
  /// c on this rank's face, shaped like a ghost slice (or a scalar tensor for a constant c)
  torch::Tensor faceValue() const;

  /// c_function at the face position and the tangential cell centres of the ghost slice
  torch::Tensor functionFaceValue(Real t) const;

  /// Robin coefficients a and b of a*u + b*du/dn = c
  const Real _a;
  const Real _b;

  /// constant c as a scalar tensor
  const torch::Tensor _c_scalar;

  /// c varying along the face, from a function or from a buffer's first owned layer at the face
  const Function * const _c_function;
  const torch::Tensor * const _c_buffer;

  /// c_function on the face, cached for the time it was evaluated at
  mutable torch::Tensor _c_function_face;
  mutable Real _c_function_time;
};
