/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMNonEquilibriumExtrapolation.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMNonEquilibriumExtrapolation);

InputParameters
LBMNonEquilibriumExtrapolation::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addClassDescription(
      "Non-equilibrium extrapolation boundary condition (Guo et al., 2002). "
      "Reconstructs the boundary distribution as f = f^eq(prescribed) + f^neq(interior), "
      "where the missing moment is extrapolated from interior nodes. "
      "Density, velocity, and equilibrium are computed on-the-fly from the streamed population.");

  MooseEnum prescribe_type("velocity density", "velocity");
  params.addParam<MooseEnum>("prescribe_type",
                             prescribe_type,
                             "Quantity prescribed at the boundary. "
                             "'velocity': prescribe ux/uy/uz and extrapolate density. "
                             "'density': prescribe rho and extrapolate velocity.");

  MooseEnum order("first second", "first");
  params.addParam<MooseEnum>(
      "order",
      order,
      "Extrapolation order for the non-prescribed quantity. "
      "'first': use one interior neighbor. "
      "'second': use two interior neighbors (requires domain >= 3 nodes per side).");

  params.addParam<std::string>("ux", "0.0", "Prescribed x-velocity (prescribe_type = velocity)");
  params.addParam<std::string>("uy", "0.0", "Prescribed y-velocity (prescribe_type = velocity)");
  params.addParam<std::string>("uz", "0.0", "Prescribed z-velocity (prescribe_type = velocity)");

  params.addParam<std::string>(
      "prescribed_rho", "1.0", "Prescribed density (prescribe_type = density)");

  return params;
}

LBMNonEquilibriumExtrapolation::LBMNonEquilibriumExtrapolation(const InputParameters & parameters)
  : LBMBoundaryCondition(parameters),
    _prescribe_type(getParam<MooseEnum>("prescribe_type")),
    _order(getParam<MooseEnum>("order") == "second" ? 2 : 1),
    _prescribed_ux(_lb_problem.getConstant<Real>(getParam<std::string>("ux"))),
    _prescribed_uy(_lb_problem.getConstant<Real>(getParam<std::string>("uy"))),
    _prescribed_uz(_lb_problem.getConstant<Real>(getParam<std::string>("uz"))),
    _prescribed_rho(_lb_problem.getConstant<Real>(getParam<std::string>("prescribed_rho")))
{
}

torch::Tensor
LBMNonEquilibriumExtrapolation::computeDensitySlice(const torch::Tensor & f_slice) const
{
  return f_slice.sum(-1, /*keepdim=*/true);
}

void
LBMNonEquilibriumExtrapolation::computeVelocitySlice(const torch::Tensor & f_slice,
                                                     const torch::Tensor & rho_slice,
                                                     torch::Tensor & ux,
                                                     torch::Tensor & uy,
                                                     torch::Tensor & uz) const
{
  ux = (f_slice * _ex).sum(-1, /*keepdim=*/true) / rho_slice;
  uy = (f_slice * _ey).sum(-1, /*keepdim=*/true) / rho_slice;
  if (_dim == 3)
    uz = (f_slice * _ez).sum(-1, /*keepdim=*/true) / rho_slice;
  else
    uz = torch::zeros_like(ux);
}

torch::Tensor
LBMNonEquilibriumExtrapolation::computeEquilibriumSlice(const torch::Tensor & rho_b,
                                                        const torch::Tensor & ux_b,
                                                        const torch::Tensor & uy_b,
                                                        const torch::Tensor & uz_b) const
{
  auto rho_4d = rho_b.dim() == 3 ? rho_b.unsqueeze(-1) : rho_b;
  auto ux_4d = ux_b.dim() == 3 ? ux_b.unsqueeze(-1) : ux_b;
  auto uy_4d = uy_b.dim() == 3 ? uy_b.unsqueeze(-1) : uy_b;
  auto uz_4d = uz_b.dim() == 3 ? uz_b.unsqueeze(-1) : uz_b;

  auto edotu = _ex * ux_4d + _ey * uy_4d + _ez * uz_4d;
  auto edotu_sqr = edotu * edotu;
  auto usqr = ux_4d * ux_4d + uy_4d * uy_4d + uz_4d * uz_4d;
  auto second_order = edotu / _lb_problem._cs2 + 0.5 * edotu_sqr / _lb_problem._cs4;
  auto third_order = 0.5 * usqr / _lb_problem._cs2;
  return _w * rho_4d * (1.0 + second_order - third_order);
}

void
LBMNonEquilibriumExtrapolation::applyNEE(int dim, int64_t b_idx, int64_t n1_idx, int64_t n2_idx)
{
  // Compute density and velocity at interior neighbor n1
  auto f_n1 = _u_owned.select(dim, n1_idx).unsqueeze(dim);
  auto rho_n1 = computeDensitySlice(f_n1);
  torch::Tensor ux_n1, uy_n1, uz_n1;
  computeVelocitySlice(f_n1, rho_n1, ux_n1, uy_n1, uz_n1);

  // Compute feq at n1
  auto feq_n1 = computeEquilibriumSlice(rho_n1, ux_n1, uy_n1, uz_n1);
  auto f_neq_n1 = f_n1 - feq_n1;

  torch::Tensor rho_b, ux_b, uy_b, uz_b;

  if (_prescribe_type == "velocity")
  {
    // Extrapolate density from the interior; prescribe velocity components.
    if (_order == 1)
      rho_b = rho_n1;
    else if (_order == 2)
    {
      auto f_n2 = _u_owned.select(dim, n2_idx).unsqueeze(dim);
      auto rho_n2 = computeDensitySlice(f_n2);
      rho_b = 2.0 * rho_n1 - rho_n2;
    }
    else
      mooseError("Invalid extrapolation order for NEE boundary condition");
    // Build prescribed velocity tensors matching the boundary slice shape.
    ux_b = torch::full_like(rho_n1, _prescribed_ux);
    uy_b = torch::full_like(rho_n1, _prescribed_uy);
    uz_b = torch::full_like(rho_n1, _prescribed_uz);
  }
  else if (_prescribe_type == "density")
  {
    // Prescribed density; extrapolate velocity from the interior.
    rho_b = torch::full_like(rho_n1, _prescribed_rho);

    if (_order == 1)
    {
      ux_b = ux_n1;
      uy_b = uy_n1;
      uz_b = uz_n1;
    }
    else if (_order == 2)
    {
      auto f_n2 = _u_owned.select(dim, n2_idx).unsqueeze(dim);
      auto rho_n2 = computeDensitySlice(f_n2);
      torch::Tensor ux_n2, uy_n2, uz_n2;
      computeVelocitySlice(f_n2, rho_n2, ux_n2, uy_n2, uz_n2);
      ux_b = 2.0 * ux_n1 - ux_n2;
      uy_b = 2.0 * uy_n1 - uy_n2;
      uz_b = 2.0 * uz_n1 - uz_n2;
    }
    else
      mooseError("Invalid extrapolation order for NEE boundary condition");
  }
  else
    mooseError("Invalid prescribe_type for NEE boundary condition");

  // Equilibrium at boundary using prescribed/extrapolated moments
  auto feq_b = computeEquilibriumSlice(rho_b, ux_b, uy_b, uz_b);

  // Non-equilibrium extrapolation
  torch::Tensor result;
  if (_order == 1)
    result = feq_b + f_neq_n1;
  else if (_order == 2)
  {
    // Compute f_neq at n2
    auto f_n2 = _u_owned.select(dim, n2_idx).unsqueeze(dim);
    auto rho_n2 = computeDensitySlice(f_n2);
    torch::Tensor ux_n2, uy_n2, uz_n2;
    computeVelocitySlice(f_n2, rho_n2, ux_n2, uy_n2, uz_n2);
    auto feq_n2 = computeEquilibriumSlice(rho_n2, ux_n2, uy_n2, uz_n2);
    result = feq_b + 2.0 * f_neq_n1 - (f_n2 - feq_n2);
  }
  else
    mooseError("Invalid extrapolation order for NEE boundary condition");

  // Write into the boundary plane of the output tensor.
  _u_owned.narrow(dim, b_idx, 1).copy_(result);
}

void
LBMNonEquilibriumExtrapolation::leftBoundary()
{
  // x = 0;  interior neighbors at x = 1 and x = 2
  applyNEE(0, 0, 1, 2);
}

void
LBMNonEquilibriumExtrapolation::rightBoundary()
{
  // x = Nx-1;  interior neighbors at Nx-2 and Nx-3
  applyNEE(0, _shape[0] - 1, _shape[0] - 2, _shape[0] - 3);
}

void
LBMNonEquilibriumExtrapolation::bottomBoundary()
{
  // y = 0;  interior neighbors at y = 1 and y = 2
  applyNEE(1, 0, 1, 2);
}

void
LBMNonEquilibriumExtrapolation::topBoundary()
{
  // y = Ny-1;  interior neighbors at Ny-2 and Ny-3
  applyNEE(1, _shape[1] - 1, _shape[1] - 2, _shape[1] - 3);
}

void
LBMNonEquilibriumExtrapolation::frontBoundary()
{
  // z = 0;  interior neighbors at z = 1 and z = 2
  applyNEE(2, 0, 1, 2);
}

void
LBMNonEquilibriumExtrapolation::backBoundary()
{
  // z = Nz-1;  interior neighbors at Nz-2 and Nz-3
  applyNEE(2, _shape[2] - 1, _shape[2] - 2, _shape[2] - 3);
}

void
LBMNonEquilibriumExtrapolation::computeBuffer()
{
  LBMBoundaryCondition::computeBuffer();
  _lb_problem.maskedFillSolids(_u_owned, 0);
}
