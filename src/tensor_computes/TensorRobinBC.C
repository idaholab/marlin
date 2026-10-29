/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorRobinBC.h"
#include "DomainAction.h"
#include "Function.h"
#include "MarlinUtils.h"

registerMooseObject("MarlinApp", TensorRobinBC);

InputParameters
TensorRobinBC::validParams()
{
  InputParameters params = TensorBoundaryCondition::validParams();
  params.addParam<Real>("a", 1, "Robin coefficient on u (a*u + b*du/dn = c).");
  params.addParam<Real>("b", 0, "Robin coefficient on du/dn (a*u + b*du/dn = c).");
  params.addParam<Real>("c", 0, "Robin right-hand-side value (a*u + b*du/dn = c).");
  params.addParam<FunctionName>(
      "c_function",
      "Function giving c along the face, evaluated at the face position at the solve time.");
  params.addParam<TensorInputBufferName>(
      "c_buffer",
      "Buffer whose first owned cell layer next to the face holds c along the face. The compute "
      "that writes it must evaluate c at the face position, not at the cell centres.");
  params.addClassDescription(
      "General Robin boundary condition a*u + b*du/dn = c on a non-periodic domain face.");
  return params;
}

TensorRobinBC::TensorRobinBC(const InputParameters & parameters)
  : TensorBoundaryCondition(parameters),
    FunctionInterface(this),
    _a(getParam<Real>("a")),
    _b(getParam<Real>("b")),
    _c_scalar(torch::tensor(getParam<Real>("c"), MooseTensor::floatTensorOptions())),
    _c_function(isParamValid("c_function") ? &getFunction("c_function") : nullptr),
    _c_buffer(isParamValid("c_buffer") ? &getInputBuffer("c_buffer") : nullptr),
    _c_function_time(std::numeric_limits<Real>::quiet_NaN())
{
  if (unsigned(isParamSetByUser("c")) + bool(_c_function) + bool(_c_buffer) > 1)
    paramError("c",
               "Give only one of ",
               parameters.checkForRename("c"),
               ", ",
               parameters.checkForRename("c_function"),
               " and ",
               parameters.checkForRename("c_buffer"),
               ".");

  // the ghost value divides by a/2 + b/dx (see RobinGhost::coefficientOnU0); this also rejects a =
  // b = 0
  const auto dx = _domain.getGridSpacing()(faceDim());
  if (std::abs(_a / 2.0 + _b / dx) <= 1e-12 * (std::abs(_a) / 2.0 + std::abs(_b) / dx))
    paramError(
        "a", "a/2 + b/dx must be nonzero (dx = ", dx, " is the grid spacing normal to the face).");
}

void
TensorRobinBC::fillGhostCellsStatic(torch::Tensor & buffer,
                                    bool is_upper_face,
                                    int64_t face_dim,
                                    unsigned int ghost_layers,
                                    int64_t halo,
                                    int64_t owned_extent,
                                    Real a,
                                    Real b,
                                    const torch::Tensor & c,
                                    Real dx)
{
  if (const auto layer = zeroDenominatorLayer(a, b, dx, ghost_layers); layer < ghost_layers)
    ::mooseError("TensorRobinBC with ",
                 ghost_layers,
                 " ghost layers requires a/2 + b/(",
                 2 * layer + 1,
                 "*dx) to be nonzero (dx = ",
                 dx,
                 " is the grid spacing normal to the face).");
  if (int64_t(ghost_layers) > owned_extent)
    ::mooseError("TensorRobinBC fills ghost layer l from owned cell l, but ",
                 ghost_layers,
                 " ghost layers were requested next to ",
                 owned_extent,
                 " owned cells.");

  // Marlin's real-space grid is cell-centered (axis built from min+dx/2 ... max-dx/2), so the
  // face sits midway between owned cell l (counted from the face) and ghost layer l, which are
  // (2l + 1) dx apart. Each layer applies the Robin relation to that pair, with the central
  // difference over the pair for du/dn. This is exact for linear fields and gives the even
  // mirror ghost_l = u_l for zero flux and the odd mirror ghost_l = 2c - u_l for Dirichlet. The
  // lower and upper face reduce to the same formula (RobinGhost::coefficientOnU0 / constantTerm)
  // because the ghost is always the cell on the outward side.
  for (const auto l : make_range(int64_t(ghost_layers)))
  {
    const auto spacing = (2.0 * l + 1.0) * dx;
    const auto u_l = is_upper_face ? buffer.narrow(face_dim, halo + owned_extent - 1 - l, 1)
                                   : buffer.narrow(face_dim, halo + l, 1);
    auto ghost_l = is_upper_face ? buffer.narrow(face_dim, halo + owned_extent + l, 1)
                                 : buffer.narrow(face_dim, halo - 1 - l, 1);
    ghost_l.copy_(RobinGhost::coefficientOnU0(a, b, spacing) * u_l +
                  c * RobinGhost::constantTerm(a, b, 1.0, spacing));
  }
}

unsigned int
TensorRobinBC::zeroDenominatorLayer(Real a, Real b, Real dx, unsigned int ghost_layers)
{
  // layer l divides by a/2 + b/((2l + 1) dx)
  for (const auto l : make_range(ghost_layers))
  {
    const auto spacing = (2.0 * l + 1.0) * dx;
    if (std::abs(a / 2.0 + b / spacing) <= 1e-12 * (std::abs(a) / 2.0 + std::abs(b) / spacing))
      return l;
  }
  return ghost_layers;
}

void
TensorRobinBC::fillGhostCells(torch::Tensor & buffer,
                              unsigned int ghost_layers,
                              int64_t halo,
                              int64_t owned_extent,
                              Real dx) const
{
  fillGhostCellsStatic(buffer,
                       isUpperFace(),
                       static_cast<int64_t>(faceDim()),
                       ghost_layers,
                       halo,
                       owned_extent,
                       _a,
                       _b,
                       faceValue(),
                       dx);
}

void
TensorRobinBC::checkGhostLayers(unsigned int ghost_layers, Real dx) const
{
  if (const auto layer = zeroDenominatorLayer(_a, _b, dx, ghost_layers); layer < ghost_layers)
    paramError("a",
               "Buffer '",
               getVariableName(),
               "' is read with ",
               ghost_layers,
               " ghost layers, and ghost layer ",
               layer + 1,
               " divides by a/2 + b/(",
               2 * layer + 1,
               "*dx), which is zero (dx = ",
               dx,
               " is the grid spacing normal to the face).");
}

void
TensorRobinBC::prepareFaceData()
{
  // the variable's ghost fill reads c across the variable's halo along the other axes
  if (_c_buffer)
    _tensor_problem.exchangeGhostLayers(getParam<TensorInputBufferName>("c_buffer"),
                                        _tensor_problem.getBufferGhostLayers(getVariableName()),
                                        faceDim());
}

torch::Tensor
TensorRobinBC::faceValue() const
{
  if (_c_buffer)
  {
    // the first owned layer next to the face, over the whole padded extent of the other axes
    // (its halo is filled by prepareFaceData)
    const auto halo = int64_t(_tensor_problem.getMaxGhostLayer());
    const auto d = faceDim();
    return _c_buffer->narrow(d, isUpperFace() ? halo + _domain.getLocalGridSize()[d] - 1 : halo, 1);
  }
  if (_c_function)
  {
    const auto t = _tensor_problem.boundaryTime();
    if (t != _c_function_time)
    {
      _c_function_face = functionFaceValue(t);
      _c_function_time = t;
    }
    return _c_function_face;
  }
  return _c_scalar;
}

torch::Tensor
TensorRobinBC::functionFaceValue(Real t) const
{
  // cell centres of the padded local grid along every axis; the face axis is set to the face
  // position. Ghost positions along the other axes are true cell positions, so later axes'
  // fills see the right c at the corners.
  std::array<int64_t, 3> begin{{0, 0, 0}}, end{{0, 0, 0}};
  _domain.getLocalBounds(_domain.comm().rank(), begin, end);
  const auto & dx = _domain.getGridSpacing();
  const auto & x0 = _domain.getDomainMin();
  const auto halo = int64_t(_tensor_problem.getMaxGhostLayer());
  const auto shape = _tensor_problem.getLocalTensorShape({});
  const auto d = faceDim();

  std::vector<int64_t> face_shape(shape.begin(), shape.end());
  face_shape[d] = 1;
  const auto position = [&](unsigned int axis, int64_t i)
  {
    if (axis == d)
      return isUpperFace() ? _domain.getDomainMax()(d) : x0(d);
    return x0(axis) + (begin[axis] + i - halo + 0.5) * dx(axis);
  };

  // row-major (last axis fastest), as the tensor below
  face_shape.resize(3, 1);
  std::vector<double> values;
  values.reserve(face_shape[0] * face_shape[1] * face_shape[2]);
  for (const auto i : make_range(face_shape[0]))
    for (const auto j : make_range(face_shape[1]))
      for (const auto k : make_range(face_shape[2]))
        values.push_back(_c_function->value(t,
                                            Point(position(0, i),
                                                  _dim > 1 ? position(1, j) : 0.0,
                                                  _dim > 2 ? position(2, k) : 0.0)));
  face_shape.resize(_dim);
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kDouble))
      .reshape(face_shape)
      .to(MooseTensor::floatTensorOptions());
}
