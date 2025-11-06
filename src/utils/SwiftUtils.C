/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SwiftUtils.h"
#include "SwiftApp.h"
#include "MooseUtils.h"
#include "Moose.h"

namespace MooseTensor
{

struct TorchDeviceSingleton
{
  static bool isSupported(torch::Dtype dtype, torch::Device device)
  {
    try
    {
      auto tensor = torch::zeros({1}, torch::dtype(dtype).device(device));
      return true;
    }
    catch (const std::exception &)
    {
      return false;
    }
  }

  TorchDeviceSingleton()
    : _device_string(torchDevice().empty() ? (torch::cuda::is_available()
                                                  ? "cuda"
                                                  : (torch::mps::is_available() ? "mps" : "cpu"))
                                           : torchDevice()),
      _device(_device_string),
      _floating_precision(precision().empty() ? "DEVICE_DEFAULT" : precision()),
      _float_dtype(_floating_precision == "DEVICE_DEFAULT" || _floating_precision == "DOUBLE"
                       ? (isSupported(torch::kFloat64, _device) ? torch::kFloat64 : torch::kFloat32)
                       : torch::kFloat32),
      _complex_float_dtype(isSupported(torch::kComplexDouble, _device) ? torch::kComplexDouble
                                                                       : torch::kComplexFloat),
      _int_dtype(isSupported(torch::kInt64, _device) ? torch::kInt64 : torch::kInt32)
  {
    mooseInfo("Running on '",
              _device_string,
              "' with ",
              _float_dtype == torch::kFloat64 ? "double" : "single",
              " precision floating point numbers");
  }

  const std::string _device_string;
  const torch::Device _device;
  const std::string _floating_precision;
  const torch::Dtype _float_dtype;
  const torch::Dtype _complex_float_dtype;
  const torch::Dtype _int_dtype;
};

void
printTensorInfo(const torch::Tensor & x)
{
  Moose::out << "      dimension: " << x.dim() << std::endl;
  Moose::out << "          shape: " << x.sizes() << std::endl;
  Moose::out << "          dtype: " << x.dtype() << std::endl;
  Moose::out << "         device: " << x.device() << std::endl;
  Moose::out << "  requires grad: " << (x.requires_grad() ? "true" : "false") << std::endl;
  Moose::out << std::endl;
}

void
printTensorInfo(const std::string & name, const torch::Tensor & x)
{
  Moose::out << "============== " << name << " ==============\n";
  printTensorInfo(x);
  Moose::out << std::endl;
}

void
printElementZero(const torch::Tensor & tensor)
{
  // Access the element at all zero indices
  auto element = tensor[0][0];
  // for (int i = 1; i < tensor.dim(); ++i)
  //   element = element[0];

  Moose::out << element << std::endl;
}

void
printElementZero(const std::string & name, const torch::Tensor & x)
{
  Moose::out << "============== " << name << " ==============\n";
  printElementZero(x);
  Moose::out << std::endl;
}

const torch::TensorOptions
floatTensorOptions()
{
  const static TorchDeviceSingleton ts;
  return torch::TensorOptions()
      .dtype(ts._float_dtype)
      .layout(torch::kStrided)
      .memory_format(torch::MemoryFormat::Contiguous)
      .pinned_memory(false)
      .device(ts._device)
      .requires_grad(false);
}

const torch::TensorOptions
complexFloatTensorOptions()
{
  const static TorchDeviceSingleton ts;
  return torch::TensorOptions()
      .dtype(ts._complex_float_dtype)
      .layout(torch::kStrided)
      .memory_format(torch::MemoryFormat::Contiguous)
      .pinned_memory(false)
      .device(ts._device)
      .requires_grad(false);
}

const torch::TensorOptions
intTensorOptions()
{
  const static TorchDeviceSingleton ts;
  return torch::TensorOptions()
      .dtype(ts._int_dtype)
      .layout(torch::kStrided)
      .memory_format(torch::MemoryFormat::Contiguous)
      .pinned_memory(false)
      .device(ts._device)
      .requires_grad(false);
}

torch::Tensor
unsqueeze0(const torch::Tensor & t, unsigned int ndim)
{
  torch::Tensor u = t;
  for (unsigned int i = 0; i < ndim; ++i)
    u = u.unsqueeze(0);
  return u;
}

torch::Tensor
trans2(const torch::Tensor & A2)
{
  return torch::einsum("...ij          ->...ji  ", {A2});
}

torch::Tensor
ddot42(const torch::Tensor & A4, const torch::Tensor & B2)
{
  return torch::einsum("...ijkl,...lk  ->...ij  ", {A4, B2});
}

torch::Tensor
ddot44(const torch::Tensor & A4, const torch::Tensor & B4)
{
  return torch::einsum("...ijkl,...lkmn->...ijmn", {A4, B4});
}

torch::Tensor
dot22(const torch::Tensor & A2, const torch::Tensor & B2)
{
  return torch::einsum("...ij  ,...jk  ->...ik  ", {A2, B2});
}

torch::Tensor
dot24(const torch::Tensor & A2, const torch::Tensor & B4)
{
  return torch::einsum("...ij  ,...jkmn->...ikmn", {A2, B4});
}

torch::Tensor
dot42(const torch::Tensor & A4, const torch::Tensor & B2)
{
  return torch::einsum("...ijkl,...lm  ->...ijkm", {A4, B2});
}

torch::Tensor
dyad22(const torch::Tensor & A2, const torch::Tensor & B2)
{
  return torch::einsum("...ij  ,...kl  ->...ijkl", {A2, B2});
}

void
printBuffer(const torch::Tensor & t, const unsigned int & precision, const unsigned int & index)
{
  /**
   * Print the entire field for debugging
   */
  torch::Tensor field = t;
  // for buffers higher than 3 dimensions, such as distribution functions
  // pass an index to print or call this method repeatedly to print all directions
  // higher than 4 dimensions is not supported

  if (t.dim() == 4)
    field = t.select(3, index);

  if (t.dim() > 4)
    mooseError("Higher than 4 dimensional tensor buffers are not supported.");

  if (t.dim() == 2)
  {
    for (int64_t j = 0; j < field.size(1); j++)
    {
      for (int64_t k = 0; k < field.size(0); k++)
        std::cout << std::fixed << std::setprecision(precision) << field[k][j].item<Real>() << " ";
      std::cout << std::endl;
    }
  }

  else if (t.dim() >= 3)
  {
    for (int64_t i = 0; i < field.size(2); i++)
    {
      for (int64_t j = 0; j < field.size(1); j++)
      {
        for (int64_t k = 0; k < field.size(0); k++)
          std::cout << std::fixed << std::setprecision(precision) << field[k][j][i].item<Real>()
                    << " ";
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }

  else if (t.dim() == 1)
  {
    for (int64_t k = 0; k < field.size(0); k++)
      std::cout << std::fixed << std::setprecision(precision) << field[k].item<Real>() << " ";
    std::cout << std::endl;
  }

  else
    mooseError("Unsupported output dimension");
}

} // namespace MooseTensor
