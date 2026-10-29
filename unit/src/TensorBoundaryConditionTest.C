/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorRobinBC.h"
#include "PetscFDHierarchy.h"
#include "MarlinUtils.h"
#include "gtest/gtest.h"

#include "libmesh/int_range.h"

#include <array>

// fillGhostCellsStatic() (the ghost fill, which the explicit computes and the implicit
// solver's fine level both use) and the ghost coefficients the implicit solver writes into
// its matrices (ghostCoefficientOnU0() in the Jacobi diagonal, PetscFD::FaceGhosts::factor()
// on the coarse multigrid levels) must, for the same (a, b, dx), imply the same dependence
// of each ghost on its mirror cell, and the ghost value must satisfy the Robin condition
// itself. The second check is independent of the shared ghostCoefficientOnU0() /
// ghostConstantTerm() helpers, so a sign error in them is caught.
namespace
{
torch::Tensor
scalar(Real c)
{
  return torch::tensor(c, MooseTensor::floatTensorOptions());
}

// Checks that the ghost fill and the implicit solver's ghost coefficients agree, and that
// the fill satisfies the BC, for one Robin triple (a, b, c), on both the lower and upper face
// of a small 1D field, for both ghost layers.
void
checkDualPath(Real a, Real b, Real c, Real dx)
{
  const int64_t halo = 2;
  const int64_t owned_extent = 4;
  const int64_t total = owned_extent + 2 * halo;
  // owned cells 0 and 1 from the face (the mirrors of ghost layers 0 and 1)
  const std::array<Real, 2> u_lower{{1.7, 0.6}}, u_upper{{-0.4, 1.1}};

  PetscFD::FaceGhosts faces;
  faces.ab[0] = faces.ab[1] = {a, b};

  // ghost values come from a tensor op (fillGhostCellsStatic), which may run in
  // single precision depending on the active compute device/dtype setting;
  // the predictions are plain double arithmetic, so the tolerance must
  // allow for that precision gap, not just floating-point roundoff.
  const Real tolerance =
      MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-12 : 1e-6;

  for (const bool is_upper_face : {false, true})
  {
    const auto & u = is_upper_face ? u_upper : u_lower;
    const auto owned = [&](int64_t l)
    { return is_upper_face ? halo + owned_extent - 1 - l : halo + l; };
    const auto ghost = [&](int64_t l)
    { return is_upper_face ? halo + owned_extent + l : halo - 1 - l; };
    const auto fill = [&](const std::array<Real, 2> & values)
    {
      auto buffer = torch::zeros({total}, MooseTensor::floatTensorOptions());
      for (const auto l : make_range(2))
        buffer[owned(l)] = values[l];
      TensorRobinBC::fillGhostCellsStatic(buffer,
                                          is_upper_face,
                                          /*face_dim=*/0,
                                          /*ghost_layers=*/2,
                                          halo,
                                          owned_extent,
                                          a,
                                          b,
                                          scalar(c),
                                          dx);
      return buffer;
    };
    const auto buffer = fill(u);
    // the ghosts of the zero field hold c's part, which the implicit solver takes from the fill
    // itself (its affine split); the rest is the homogeneous part it writes into its matrices
    const auto offset = fill({{0.0, 0.0}});

    for (const auto l : make_range(2))
    {
      const auto homogeneous = buffer[ghost(l)].item<double>() - offset[ghost(l)].item<double>();
      EXPECT_NEAR(homogeneous, faces.factor(is_upper_face, l, dx) * u[l], tolerance)
          << "a=" << a << " b=" << b << " c=" << c << " dx=" << dx
          << " is_upper_face=" << is_upper_face << " layer=" << l;
      if (l == 0)
        EXPECT_NEAR(homogeneous, TensorRobinBC::ghostCoefficientOnU0(a, b, dx) * u[0], tolerance);
    }

    const auto u0 = u[0];
    const auto ghost_value = buffer[ghost(0)].item<double>();

    // the ghost satisfies a*u + b*du/dn = c at the face: u = (u0 + ughost)/2 and,
    // since the ghost lies on the outward side of either face, du/dn = (ughost - u0)/dx
    const Real bc_residual = a * (u0 + ghost_value) / 2.0 + b * (ghost_value - u0) / dx - c;
    EXPECT_NEAR(bc_residual, 0.0, tolerance * (std::abs(a) / 2.0 + std::abs(b) / dx))
        << "a=" << a << " b=" << b << " c=" << c << " dx=" << dx
        << " is_upper_face=" << is_upper_face;
  }
}

// Checks that the second ghost layer satisfies the Robin condition on the pair (u1, ghost2),
// which is centred on the face and 3*dx apart, on both faces.
void
checkSecondGhostLayer(Real a, Real b, Real c, Real dx)
{
  const int64_t halo = 2;
  const int64_t owned_extent = 4;
  const Real u1 = 0.9; // second cell in from either face
  const Real tolerance =
      MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-12 : 1e-6;

  for (const bool is_upper_face : {false, true})
  {
    auto buffer = torch::zeros({owned_extent + 2 * halo}, MooseTensor::floatTensorOptions());
    buffer[is_upper_face ? halo + owned_extent - 2 : halo + 1] = u1;

    TensorRobinBC::fillGhostCellsStatic(buffer,
                                        is_upper_face,
                                        /*face_dim=*/0,
                                        /*ghost_layers=*/2,
                                        halo,
                                        owned_extent,
                                        a,
                                        b,
                                        scalar(c),
                                        dx);

    const auto ghost2 = buffer[is_upper_face ? halo + owned_extent + 1 : halo - 2].item<double>();
    const Real bc_residual = a * (u1 + ghost2) / 2.0 + b * (ghost2 - u1) / (3.0 * dx) - c;
    EXPECT_NEAR(bc_residual, 0.0, tolerance * (std::abs(a) / 2.0 + std::abs(b) / dx))
        << "a=" << a << " b=" << b << " c=" << c << " dx=" << dx
        << " is_upper_face=" << is_upper_face;
  }
}
}

TEST(TensorBoundaryConditionTest, DirichletDualPath)
{
  // a=1, b=0 (TensorDirichletBC preset): u = value on the face.
  checkDualPath(/*a=*/1.0, /*b=*/0.0, /*c=*/3.5, /*dx=*/0.1);
}

TEST(TensorBoundaryConditionTest, NeumannDualPath)
{
  // a=0, b=1 (TensorNeumannBC preset): du/dn = flux on the face.
  checkDualPath(/*a=*/0.0, /*b=*/1.0, /*c=*/-2.0, /*dx=*/0.1);
}

TEST(TensorBoundaryConditionTest, GeneralRobinDualPath)
{
  checkDualPath(/*a=*/2.0, /*b=*/3.0, /*c=*/1.25, /*dx=*/0.05);
}

TEST(TensorBoundaryConditionTest, SecondGhostLayer)
{
  checkSecondGhostLayer(/*a=*/1.0, /*b=*/0.0, /*c=*/3.5, /*dx=*/0.1);
  checkSecondGhostLayer(/*a=*/0.0, /*b=*/1.0, /*c=*/-2.0, /*dx=*/0.1);
  checkSecondGhostLayer(/*a=*/2.0, /*b=*/3.0, /*c=*/1.25, /*dx=*/0.05);
}

TEST(TensorBoundaryConditionTest, DeepGhostLayers)
{
  // three layers (a 7-point stencil): layer l and owned cell l are (2l + 1) dx apart, centred on
  // the face; Dirichlet gives the odd mirror about c, Neumann the even mirror plus the flux
  const int64_t halo = 3;
  const int64_t owned_extent = 4;
  const Real dx = 0.1;
  const Real tolerance =
      MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-12 : 1e-5;
  const std::vector<Real> u{0.3, -1.1, 0.7, 2.0};
  for (const auto & [a, b, c] :
       std::vector<std::array<Real, 3>>{{1.0, 0.0, 3.5}, {0.0, 1.0, -2.0}, {2.0, 3.0, 1.25}})
    for (const bool is_upper_face : {false, true})
    {
      auto buffer = torch::zeros({owned_extent + 2 * halo}, MooseTensor::floatTensorOptions());
      for (const auto i : index_range(u))
        buffer[halo + int64_t(i)] = u[i];
      TensorRobinBC::fillGhostCellsStatic(
          buffer, is_upper_face, /*face_dim=*/0, 3, halo, owned_extent, a, b, scalar(c), dx);
      for (const auto l : make_range(int64_t(3)))
      {
        const auto u_l = u[is_upper_face ? owned_extent - 1 - l : l];
        const auto ghost =
            buffer[is_upper_face ? halo + owned_extent + l : halo - 1 - l].item<double>();
        const auto spacing = (2.0 * l + 1.0) * dx;
        EXPECT_NEAR(a * (u_l + ghost) / 2.0 + b * (ghost - u_l) / spacing - c,
                    0.0,
                    tolerance * (std::abs(a) / 2.0 + std::abs(b) / dx))
            << "a=" << a << " b=" << b << " layer=" << l << " is_upper_face=" << is_upper_face;
      }
    }
}

TEST(TensorBoundaryConditionTest, MoreGhostLayersThanOwnedCellsIsAnError)
{
  auto buffer = torch::zeros({10}, MooseTensor::floatTensorOptions());
  EXPECT_THROW(TensorRobinBC::fillGhostCellsStatic(buffer,
                                                   /*is_upper_face=*/false,
                                                   /*face_dim=*/0,
                                                   /*ghost_layers=*/3,
                                                   /*halo=*/3,
                                                   /*owned_extent=*/2,
                                                   1.0,
                                                   0.0,
                                                   scalar(0.0),
                                                   0.1),
               std::exception);
}

TEST(TensorBoundaryConditionTest, ZeroThirdLayerDenominator)
{
  // a/2 + b/(5dx) = 0 with b = 1, dx = 0.1; the first two layers' denominators are nonzero, so the
  // third layer is reported, and with two layers none is
  EXPECT_EQ(TensorRobinBC::zeroDenominatorLayer(-4.0, 1.0, 0.1, 3), 2u);
  EXPECT_EQ(TensorRobinBC::zeroDenominatorLayer(-4.0, 1.0, 0.1, 2), 2u);
}

TEST(TensorBoundaryConditionTest, ZeroSecondLayerDenominatorIsAnError)
{
  // a/2 + b/(3dx) = 0 with b = 1, dx = 0.1; a/2 + b/dx is nonzero
  auto buffer = torch::zeros({8}, MooseTensor::floatTensorOptions());
  EXPECT_THROW(TensorRobinBC::fillGhostCellsStatic(buffer,
                                                   /*is_upper_face=*/false,
                                                   /*face_dim=*/0,
                                                   /*ghost_layers=*/2,
                                                   /*halo=*/2,
                                                   /*owned_extent=*/4,
                                                   -2.0 / 0.3,
                                                   1.0,
                                                   scalar(0.0),
                                                   0.1),
               std::exception);
}

TEST(TensorBoundaryConditionTest, FaceVaryingValue)
{
  // a c that varies along a 2D face (axis 1) gives each ghost cell its own Robin relation
  const int64_t halo = 2, owned_extent = 4, tangential = 5;
  const Real a = 2.0, b = 3.0, dx = 0.05;
  const Real tolerance =
      MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-12 : 1e-5;
  const auto options = MooseTensor::floatTensorOptions();

  for (const bool is_upper_face : {false, true})
  {
    auto buffer = torch::zeros({owned_extent + 2 * halo, tangential}, options);
    const auto u0_index = is_upper_face ? halo + owned_extent - 1 : halo;
    const auto u0 = torch::linspace(-1.0, 2.0, tangential, options);
    buffer[u0_index] = u0;
    const auto c = torch::linspace(0.5, -3.0, tangential, options).unsqueeze(0);

    TensorRobinBC::fillGhostCellsStatic(
        buffer, is_upper_face, /*face_dim=*/0, /*ghost_layers=*/1, halo, owned_extent, a, b, c, dx);

    const auto ghost = buffer[is_upper_face ? halo + owned_extent : halo - 1];
    const auto residual = a * (u0 + ghost) / 2.0 + b * (ghost - u0) / dx - c.squeeze(0);
    EXPECT_LT(residual.abs().max().item<double>(), tolerance * (std::abs(a) / 2.0 + b / dx))
        << "is_upper_face=" << is_upper_face;
  }
}
