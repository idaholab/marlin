/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMIsotropicGradient.h"

using namespace torch::indexing;

registerMooseObject("MarlinApp", LBMIsotropicGradient);

InputParameters
LBMIsotropicGradient::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute isotropic gradient object.");
  params.addRequiredParam<TensorInputBufferName>("scalar_field",
                                                 "Scalar field to compute the gradient of");

  return params;
}

LBMIsotropicGradient::LBMIsotropicGradient(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters), _scalar_field(getInputBuffer("scalar_field", _radius))
{
  const unsigned int & dim = _domain.getDim();

  // Note: if D3Q19 stencil is used, isotropic gradient is NOT going to work,
  // because D3Q19 is NOT isotropic.

  if (_stencil._q == 19)
    mooseError("Isotropic gradient cannot be computed for D3Q19 stencil");

  switch (dim)
  {
    case 3:
    {
      _kernel = torch::zeros({3, 3, 3, (int64_t)dim}, MooseTensor::floatTensorOptions());
      auto kernel_of_kernel =
          torch::index_select(_stencil._weights, 0, _stencil._reorder_indices).reshape({3, 3, 3});
      auto ex3x3x3 =
          torch::index_select(_stencil._ex, 0, _stencil._reorder_indices).reshape({3, 3, 3});
      auto ey3x3x3 =
          torch::index_select(_stencil._ey, 0, _stencil._reorder_indices).reshape({3, 3, 3});
      auto ez3x3x3 =
          torch::index_select(_stencil._ez, 0, _stencil._reorder_indices).reshape({3, 3, 3});

      _kernel.index_put_({Slice(), Slice(), Slice(), 0}, kernel_of_kernel * ex3x3x3);
      _kernel.index_put_({Slice(), Slice(), Slice(), 1}, kernel_of_kernel * ey3x3x3);
      _kernel.index_put_({Slice(), Slice(), Slice(), 2}, kernel_of_kernel * ez3x3x3);

      break;
    }
    case 2:
    {
      _kernel = torch::zeros({3, 3, (int64_t)dim}, MooseTensor::floatTensorOptions());
      auto kernel_of_kernel =
          torch::index_select(_stencil._weights, 0, _stencil._reorder_indices).reshape({3, 3});
      auto ex3x3 = torch::index_select(_stencil._ex, 0, _stencil._reorder_indices).reshape({3, 3});
      auto ey3x3 = torch::index_select(_stencil._ey, 0, _stencil._reorder_indices).reshape({3, 3});

      _kernel.index_put_({Slice(), Slice(), 0}, kernel_of_kernel * ex3x3);
      _kernel.index_put_({Slice(), Slice(), 1}, kernel_of_kernel * ey3x3);

      _conv2d_options.bias(torch::Tensor()).stride({1, 1}).padding(0);
      break;
    }
  }

  // stencil offsets in kernel (reordered) order as host values, for the 3D shifted views
  if (dim == 3)
  {
    const auto ex_r = torch::index_select(_stencil._ex, 0, _stencil._reorder_indices).cpu();
    const auto ey_r = torch::index_select(_stencil._ey, 0, _stencil._reorder_indices).cpu();
    const auto ez_r = torch::index_select(_stencil._ez, 0, _stencil._reorder_indices).cpu();
    for (int64_t m = 0; m < ex_r.size(0); m++)
      _offsets.push_back(
          {ex_r[m].item<int64_t>(), ey_r[m].item<int64_t>(), ez_r[m].item<int64_t>()});
  }

  // determine the position of partition
  const auto real_space_index = _domain.getRealSpaceIndex();
  const auto real_space_parts = _domain.getRealSpacePartitions();

  _is_interior = true;
  for (unsigned int d = 0; d < dim; d++)
    if (real_space_index[d] == 0 || real_space_index[d] == real_space_parts[d] - 1)
      _is_interior = false;
}

torch::Tensor
LBMIsotropicGradient::scalarField() const
{
  // 2D scalar buffers have a trailing dim of 1 that must be removed for conv2d (use a view, the
  // input buffer is shared with other objects)
  if (_domain.getDim() == 2 && _scalar_field.dim() > 2)
    return _scalar_field.squeeze(-1);
  return _scalar_field;
}

torch::Tensor
LBMIsotropicGradient::padScalarField(const torch::Tensor & field) const
{
  // replicate-edge padding of every dimension in one allocation (instead of one cat per dim)
  namespace F = torch::nn::functional;
  std::vector<int64_t> pad(2 * field.dim(), _padding);
  auto batched = field;
  for (int i = 0; i < 2; i++)
    batched = batched.unsqueeze(0);
  return F::pad(batched, F::PadFuncOptions(pad).mode(torch::kReplicate)).squeeze(0).squeeze(0);
}

torch::Tensor
LBMIsotropicGradient::prepareInputField()
{
  const unsigned int dim = _domain.getDim();

  const auto scalar_field = scalarField();

  // Serial: replicate-edge pad on all sides
  if (_domain.comm().size() == 1)
    return padScalarField(scalar_field);

  // Parallel boundary partitions: replicate ghost edges
  if (!_is_interior)
  {
    const auto & idx = _domain.getRealSpaceIndex();
    const auto & parts = _domain.getRealSpacePartitions();
    auto field = scalar_field.clone();

    for (unsigned int d = 0; d < dim; d++)
    {
      std::vector<TensorIndex> lo(dim, Slice()), lo_src(dim, Slice());
      std::vector<TensorIndex> hi(dim, Slice()), hi_src(dim, Slice());
      lo[d] = (int64_t)0;
      lo_src[d] = (int64_t)1;
      hi[d] = (int64_t)-1;
      hi_src[d] = (int64_t)-2;

      if (idx[d] == 0)
        field.index_put_(lo, field.index(lo_src));
      if (idx[d] == parts[d] - 1)
        field.index_put_(hi, field.index(hi_src));
    }
    return field;
  }

  // Interior partition: ghost data already exchanged
  return scalar_field;
}

torch::Tensor
LBMIsotropicGradient::stencilConvolve3D(const torch::Tensor & field,
                                        const torch::Tensor & kernel_flat) const
{
  // field:        [Nx+2p, Ny+2p, Nz+2p]  (padded / ghost-layer field)
  // kernel_flat:  [Q, out_channels]       (per-direction weights for each output channel)
  // returns:      [out_channels, Nx, Ny, Nz]

  const int64_t p = _padding;
  const int64_t Nx = field.size(0) - 2 * p;
  const int64_t Ny = field.size(1) - 2 * p;
  const int64_t Nz = field.size(2) - 2 * p;
  const int64_t C = kernel_flat.size(1);
  const int64_t Q = kernel_flat.size(0);

  auto result = torch::zeros({C, Nx, Ny, Nz}, field.options());
  auto product = torch::empty_like(result);

  for (int64_t m = 0; m < Q; m++)
  {
    // the field is padded, so the shifted interior is just a view (no roll / copy needed)
    const auto & [ex, ey, ez] = _offsets[m];
    auto interior = field.narrow(0, p + ex, Nx).narrow(1, p + ey, Ny).narrow(2, p + ez, Nz);
    torch::mul_out(product, kernel_flat.select(0, m).view({C, 1, 1, 1}), interior.unsqueeze(0));
    result.add_(product);
  }

  return result;
}

void
LBMIsotropicGradient::computeBuffer()
{
  if ((unsigned int)_u.size(-1) != _domain.getDim())
    mooseError("Output buffer must have the same number of dimensions as the domain.");

  const unsigned int dim = _domain.getDim();
  _lb_problem.exchangeGhostLayers(getParam<TensorInputBufferName>("scalar_field"), _radius);

  torch::Tensor input_field = prepareInputField().unsqueeze(0).unsqueeze(0);

  // Convolve
  torch::Tensor result;
  if (dim == 3)
  {
    result = stencilConvolve3D(input_field.squeeze(0).squeeze(0),
                               _kernel.reshape({_stencil._q, (int64_t)dim}));
  }
  else
  {
    auto kernel = _kernel.permute({2, 0, 1}).unsqueeze(1);
    result = torch::nn::functional::conv2d(input_field, kernel, _conv2d_options);
  }

  // result: [1, dim, Nx, Ny(, Nz)] -> [dim, Nx, Ny(, Nz)]
  result = result.squeeze(0);

  _u_owned = ownedView(_u);
  for (unsigned int d = 0; d < dim; d++)
  {
    auto component = result.select(0, d);
    if (dim == 2)
      component = component.unsqueeze(-1);
    _u_owned.index_put_({Slice(), Slice(), Slice(), (int64_t)d}, component);
  }
  _u_owned.div_(_lb_problem._cs2);
  _lb_problem.maskedFillSolids(_u_owned, _solid_values);
}
