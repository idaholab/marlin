/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMBoundaryCondition.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

InputParameters
LBMBoundaryCondition::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  MooseEnum boundary("top bottom left right front back wall regional");
  params.addRequiredParam<MooseEnum>(
      "boundary", boundary, "Edges/Faces where boundary condition is applied.");
  params.addClassDescription("LBMBoundaryCondition object.");
  return params;
}

LBMBoundaryCondition::LBMBoundaryCondition(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _boundary(getParam<MooseEnum>("boundary").getEnum<Boundary>())
{

  auto r = _domain.comm().rank();
  std::array<int64_t, 3> begin, end;
  _domain.getLocalBounds(r, begin, end);
  auto n_global = _domain.getGridSize();

  if (end[1] == n_global[1])
    _boundary_rank |= (1 << 0); // Mark as TOP
  if (begin[1] == 0)
    _boundary_rank |= (1 << 1); // Mark as BOTTOM
  if (begin[0] == 0)
    _boundary_rank |= (1 << 2); // Mark as LEFT
  if (end[0] == n_global[0])
    _boundary_rank |= (1 << 3); // Mark as RIGHT
  if (begin[2] == 0)
    _boundary_rank |= (1 << 4); // Mark as FRONT
  if (end[2] == n_global[2])
    _boundary_rank |= (1 << 5); // Mark as BACK

  // for binary media
  if (_lb_problem.isBinaryMedia())
  {
    const torch::Tensor & binary_mesh = _lb_problem.getBinaryMedia();
    _binary_mesh = binary_mesh.clone();
    // mark 6 (64 in decimal) for wall boundary ownership
    if (isBoundaryOwned(0))
      _boundary_rank |= (1 << 6);
  }
}

void
LBMBoundaryCondition::maskBoundary()
{
  // If rank > 0, we have ghost layers, so we use 0 padding (consume the ghosts)
  int p = (_domain.comm().size() > 1) ? 0 : 1;

  auto zeros_mask = (_binary_mesh == 0).to(MooseTensor::floatTensorOptions());
  zeros_mask = zeros_mask.unsqueeze(0).unsqueeze(0);

  torch::Tensor has_zero_neighbor;
  if (_domain.getDim() == 2)
  {
    has_zero_neighbor = torch::max_pool3d(zeros_mask, {3, 3, 1}, {1, 1, 1}, {p, p, 0});
  }
  else if (_domain.getDim() == 3)
  {
    has_zero_neighbor = torch::max_pool3d(zeros_mask, {3, 3, 3}, {1, 1, 1}, {p, p, p});
  }
  else
  {
    mooseError("Domain dimension not supported.");
  }

  has_zero_neighbor = has_zero_neighbor.squeeze(0).squeeze(0);

  // Align the views
  auto owned_mesh = (_domain.comm().size() > 1) ? ownedView(_binary_mesh) : _binary_mesh;

  auto boundary_mask = (owned_mesh == 1) & (has_zero_neighbor > 0);
  owned_mesh.copy_(torch::where(boundary_mask, torch::full_like(owned_mesh, -1), owned_mesh));
}

bool
LBMBoundaryCondition::isBoundaryOwned(const int & value)
{
  if (!_lb_problem.isBinaryMedia())
    return false;

  return torch::any(_binary_mesh == value).item<bool>();
}

void
LBMBoundaryCondition::computeBuffer()
{
  // Cast the enum to the underlying bit value for the check
  const uint8_t bit = (1 << static_cast<int>(_boundary));

  if (!(_boundary_rank & bit))
    return;

  switch (_boundary)
  {
    case Boundary::left:
      leftBoundary();
      break;
    case Boundary::right:
      rightBoundary();
      break;
    case Boundary::bottom:
      bottomBoundary();
      break;
    case Boundary::top:
      topBoundary();
      break;
    case Boundary::front:
      frontBoundary();
      break;
    case Boundary::back:
      backBoundary();
      break;
    case Boundary::wall:
      wallBoundary();
      break;
    case Boundary::regional:
      regionalBoundary();
      break;
    default:
      mooseError("Undefined boundary names");
  }
}
