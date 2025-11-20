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

  // reorder indices to cosntruct square/cube
  torch::Tensor _reorder_indices;
};
