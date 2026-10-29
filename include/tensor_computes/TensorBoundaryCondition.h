/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperatorBase.h"

/**
 * Base class for boundary conditions on a real-space TensorBuffer field at a
 * non-periodic physical edge of the domain. Unlike a normal TensorOperator, a
 * boundary condition is not run by the ordinary compute list -- it is looked
 * up by TensorProblem::exchangeGhostLayers() for the specific (variable, face)
 * it acts on and invoked on demand to fill ghost cells.
 */
class TensorBoundaryCondition : public TensorOperatorBase
{
public:
  static InputParameters validParams();

  TensorBoundaryCondition(const InputParameters & parameters);

  /// which face of the domain this BC instance is attached to
  enum class Face : unsigned int
  {
    LEFT = 0,   // -X
    RIGHT = 1,  // +X
    BOTTOM = 2, // -Y
    TOP = 3,    // +Y
    FRONT = 4,  // -Z
    BACK = 5    // +Z
  };

  Face getFace() const { return _face; }

  /// spatial dimension index (0=X, 1=Y, 2=Z) that _face lies along
  unsigned int faceDim() const { return static_cast<unsigned int>(_face) / 2; }

  /// true if _face is the upper (+) side of faceDim(), false if lower (-)
  bool isUpperFace() const { return static_cast<unsigned int>(_face) % 2 == 1; }

  /// name of the TensorBuffer field this BC applies to
  const TensorInputBufferName & getVariableName() const { return _variable_name; }

  /// input-file name of a face ("left", "right", ...)
  static std::string faceName(unsigned int face);

  /**
   * Fill the ghost layer(s) for this BC's variable on this rank's side of
   * its face, in-place in the padded tensor `buffer`, using `ghost_layers`
   * ghost cells, halo padding `halo`, local owned extent `owned_extent`
   * along faceDim(), and grid spacing `dx` along faceDim(). This is the
   * explicit / matrix-free path consumed by
   * TensorProblem::exchangeGhostLayers().
   */
  virtual void fillGhostCells(torch::Tensor & buffer,
                              unsigned int ghost_layers,
                              int64_t halo,
                              int64_t owned_extent,
                              Real dx) const = 0;

  /// update data the ghost fill reads on the face (e.g. exchange a buffer's halo); called on
  /// every rank before the variable's ghost exchange
  virtual void prepareFaceData() {}

  /// error at setup if fillGhostCells() cannot fill ghost_layers layers at the spacing dx
  virtual void checkGhostLayers(unsigned int /*ghost_layers*/, Real /*dx*/) const {}

  /// Boundary conditions are not driven by the normal compute list.
  virtual void computeBuffer() override {}

protected:
  /// the face this BC instance is attached to
  const Face _face;

  /// the field name this BC targets
  const TensorInputBufferName _variable_name;
};
