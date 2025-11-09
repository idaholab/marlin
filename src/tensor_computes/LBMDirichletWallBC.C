
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMDirichletWallBC.h"

using namespace torch::indexing;

registerMooseObject("SwiftApp", LBMDirichletWallBC);

InputParameters
LBMDirichletWallBC::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addRequiredParam<TensorInputBufferName>("f_old", "Old state distribution function");
  params.addClassDescription("LBMDirichletWallBC object");
  params.addRequiredParam<TensorInputBufferName>("velocity", "Fluid velocity");
  params.addParam<Real>("value",
                        "0.0"
                        "Value at the boundary");
  params.addParam<TensorInputBufferName>("value_tensor", "Value at the boundary");
  return params;
}

LBMDirichletWallBC::LBMDirichletWallBC(const InputParameters & parameters)
  : LBMBoundaryCondition(parameters),
    _f_old(_lb_problem.getBufferOld(getParam<TensorInputBufferName>("f_old"), 1)),
    _velocity(getInputBuffer("velocity")),
    _is_value_tensor(isParamValid("value_tensor"))
{
  computeBoundaryNormals();
}

void
LBMDirichletWallBC::computeBoundaryNormals()
{
  if (_lb_problem.isBinaryMedia())
  {
    const torch::Tensor & binary_media = _lb_problem.getBinaryMedia();
    _binary_mesh = binary_media.clone();

    /* boolean travelling tensor will indicate which directions at every boundary lattice need to be
     * computed. This is done by rolling binary media around in each direction and finding where the
     * zeros are. */

    torch::Tensor boolean_travelling_tensor =
        torch::ones(_shape_q, MooseTensor::intTensorOptions());

    for (int64_t ic = 1; ic < _stencil._q; ic++)
    {
      int64_t ex = _stencil._ex[ic].item<int64_t>();
      int64_t ey = _stencil._ey[ic].item<int64_t>();
      int64_t ez = _stencil._ez[ic].item<int64_t>();
      torch::Tensor shifted_mesh = torch::roll(binary_media, {ex, ey, ez}, {0, 1, 2});
      torch::Tensor adjacent_to_boundary = (shifted_mesh == 0) & (binary_media == 1);
      auto boolean_travelling_tensor_slice = torch::ones_like(adjacent_to_boundary);
      boolean_travelling_tensor_slice.masked_fill_(adjacent_to_boundary, 0);
      boolean_travelling_tensor.index_put_({Slice(), Slice(), Slice(), ic},
                                           boolean_travelling_tensor_slice);
      _binary_mesh.masked_fill_(adjacent_to_boundary, 2);
    }

    /* normal vectors are computed by summing up microscopic velocity vectors e_i at boundaries
     * where these vectors are pointing away from the boundary */

    // boolean_travelling_tensor = 1 - boolean_travelling_tensor;
    // boolean_travelling_tensor.unsqueeze_(-1);
    // torch::Tensor e_xyz = torch::stack({_ex, _ey, _ez}).permute({1, 2, 3, 4, 0});

    // torch::Tensor normals = torch::einsum("abcqm,ijkqn->abcn", {boolean_travelling_tensor,
    // e_xyz})
    //                             .to(MooseTensor::floatTensorOptions());

    // _boundary_normals = normals / torch::norm(normals, 2, -1).unsqueeze(-1);
    // _boundary_normals = torch::where(
    //     torch::isnan(_boundary_normals), torch::zeros_like(_boundary_normals),
    //     _boundary_normals);

    // /* tangent vectors simply t_i = e_i - (e_i dot n) * n where n is unit normal vector to the
    //  * boundary */

    // _e_xyz = e_xyz.to(MooseTensor::floatTensorOptions());
    // _boundary_tangent_vectors =
    //     _e_xyz - torch::einsum("ijkqm,abcdm->abcq", {_e_xyz, _boundary_normals.unsqueeze(-2)})
    //                      .unsqueeze(-1) *
    //                  _boundary_normals.unsqueeze(-2);
  }
  else
    mooseError("Binary media must be avialble to use LBMDirichletWallBC boundary condition.");
}

void
LBMDirichletWallBC::wallBoundary()
{

  if (_lb_problem.getTotalSteps() == 0)
  {
    _boundary_mask = (_binary_mesh.unsqueeze(-1).expand_as(_u) == 2);
    _boundary_mask = _boundary_mask.to(torch::kBool);
  }

  torch::Tensor f_bounce_back;
  if (_is_value_tensor)
  {
    _value_tensor = getInputBuffer("value_tensor");
    f_bounce_back = torch::ones_like(_u) * _w * _value_tensor.unsqueeze(-1);
  }
  else
  {
    _value = getParam<Real>("value");
    f_bounce_back = torch::ones_like(_u) * _w * _value;
  }

  _u.index_put_({_boundary_mask}, f_bounce_back.index({_boundary_mask}));
}
