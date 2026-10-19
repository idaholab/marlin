/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "torch/torch.h"

#include "MooseObject.h"

#include <optional>
#include "MarlinTypes.h"
#include "MarlinUtils.h"

/**
 * Base class for lattice stencils
 */

class LatticeBoltzmannStencilBase : public MooseObject
{
public:
  static InputParameters validParams();

  LatticeBoltzmannStencilBase(const InputParameters & parameters);

public:
  int _q;
  torch::Tensor _ex;
  torch::Tensor _ey;
  torch::Tensor _ez;
  torch::Tensor _weights;
  torch::Tensor _M;
  torch::Tensor _M_inv;
  torch::Tensor _S;

  // incoming unknown distribution functions at every face
  torch::Tensor _top;
  torch::Tensor _bottom;
  torch::Tensor _left;
  torch::Tensor _right;
  torch::Tensor _front;
  torch::Tensor _back;

  // indices of opposite directions
  torch::Tensor _op;

  // further elaborate boundaries
  torch::Tensor _neutral_x;       // the ones with ex = 0
  torch::Tensor _neutral_x_pos_y; // the ones with ex = 0 and +ey
  torch::Tensor _neutral_x_neg_y; // the ones with ex = 0 and -ey
  torch::Tensor _neutral_x_pos_z; // the ones with ex = 0 and +ez
  torch::Tensor _neutral_x_neg_z; // the ones with ex = 0 and -ez

  torch::Tensor _neutral_y; // the ones with ey = 0
  torch::Tensor _neutral_z; // // the ones with ez = 0

  // indices of shear viscosity relaxation
  torch::Tensor _id_kinematic_visc;

  // reorder indices to construct square/cube
  torch::Tensor _reorder_indices;

  /// host (CPU) copies of the stencil tables, for scalar lookups in per-step loops
  /// (item() on a device tensor is a host-device synchronization)
  struct HostTables
  {
    std::vector<int64_t> ex, ey, ez, op, left, right, bottom, top, front, back;
    std::vector<Real> weights;
  };

  /// host tables, built on first use (after the derived stencil has filled the tensors)
  const HostTables & host() const;

private:
  mutable std::optional<HostTables> _host;
};
