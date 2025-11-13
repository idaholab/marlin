/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMTensorBuffer.h"
#include "DomainAction.h"
#include "LatticeBoltzmannStencilBase.h"
#include "LatticeBoltzmannProblem.h"

#ifdef LIBMESH_HAVE_HDF5
#include "hdf5.h"
#endif

registerMooseObject("MarlinApp", LBMTensorBuffer);

InputParameters
LBMTensorBuffer::validParams()
{
  InputParameters params = TensorBuffer<torch::Tensor>::validParams();
  params.addRequiredParam<std::string>("buffer_type",
                                       "The buffer type can be either distribution function (df), "
                                       "macroscopic scalar (ms) or macroscopic vectorial (mv)");

  params.addParam<FileName>("file", "Optional path of the file to read tensor form.");

  params.addParam<bool>("is_integer", false, "Whether to specify integer dtype");
  params.addPrivateParam<TensorProblem *>("_tensor_problem", nullptr);
  params.addClassDescription("Tensor wrapper form LBM tensors");

  return params;
}

LBMTensorBuffer::LBMTensorBuffer(const InputParameters & parameters)
  : TensorBuffer<torch::Tensor>(parameters),
    _buffer_type(getParam<std::string>("buffer_type")),
    _lb_problem(dynamic_cast<LatticeBoltzmannProblem &>(
        *getCheckedPointerParam<TensorProblem *>("_tensor_problem"))),
    _stencil(_lb_problem.getStencil())
{
}

void
LBMTensorBuffer::init()
{
  int64_t dimension = 0;
  if (_buffer_type == "df")
    dimension = _stencil._q;
  else if (_buffer_type == "mv")
    dimension = _domain.getDim();
  else if (_buffer_type == "ms")
    dimension = 0;
  else
    mooseError("Buffer type ", _buffer_type, " is not recognized");

  std::vector<int64_t> shape(_domain.getShape().begin(), _domain.getShape().end());

  if (_domain.getDim() < 3)
    shape.push_back(1);
  if (dimension > 0)
    shape.push_back(static_cast<int64_t>(dimension));

  if (getParam<bool>("is_integer"))
    _u = torch::zeros(shape, MooseTensor::intTensorOptions());
  else
    _u = torch::zeros(shape, MooseTensor::floatTensorOptions());

  if (isParamValid("file"))
    readTensorFromHdf5();
}

void
LBMTensorBuffer::readTensorFromFile(const std::vector<int64_t> & shape)
{
  mooseDeprecated("readTensorFromFile is deprecated, use h5 reader readTensorFromHdf5 instead!");

  const FileName tensor_file = getParam<FileName>("file");
  mooseInfo("Loading tensor(s) from file \n" + tensor_file);
  std::ifstream file(tensor_file);
  if (!file.is_open())
    mooseError("Cannot open file " + tensor_file);

  // read file into standart vector
  std::vector<Real> fileData(shape[0] * shape[1] * shape[2]);

  for (unsigned int i = 0; i < fileData.size(); i++)
    if (!(file >> fileData[i]))
      mooseError("Insufficient data in the file");

  file.close();

  // reshape and write into torch tensor
  for (int64_t k = 0; k < shape[2]; k++)
    for (int64_t j = 0; j < shape[1]; j++)
      for (int64_t i = 0; i < shape[0]; i++)
      {
        if (getParam<bool>("is_integer"))
          _u.index_put_({i, j, k},
                        static_cast<int>(fileData[k * shape[1] * shape[0] + j * shape[0] + i]));
        else
          _u.index_put_({i, j, k}, fileData[k * shape[1] * shape[0] + j * shape[0] + i]);
      }
}

void
LBMTensorBuffer::readTensorFromHdf5()
{
#ifdef LIBMESH_HAVE_HDF5
  const FileName tensor_file_name = getParam<FileName>("file");

  auto tensor_file_char = tensor_file_name.c_str();

  // open file
  hid_t file_id = H5Fopen(tensor_file_char, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0)
    mooseError("Failed to open h5 file");

  std::string dataset_name = tensor_file_name.substr(0, tensor_file_name.size() - 3);
  auto last_slash = dataset_name.find_last_of("/\\");
  if (last_slash != std::string::npos)
    dataset_name = dataset_name.substr(last_slash + 1);
  auto dataset_name_char = dataset_name.c_str();

  // open dataset
  hid_t dataset_id = H5Dopen2(file_id, dataset_name_char, H5P_DEFAULT);
  if (dataset_id < 0)
    mooseError("Failed to obtain dataset from h5 file");

  // get dataspace
  hid_t dataspace_id = H5Dget_space(dataset_id);
  if (dataspace_id < 0)
    mooseError("Failed to obtain dataspace from h5 dataset");

  // get the dimensions of the dataspace
  const hsize_t rank = H5Sget_simple_extent_ndims(dataspace_id);
  std::vector<hsize_t> dims(rank);
  H5Sget_simple_extent_dims(dataspace_id, dims.data(), NULL);

  // get memory type id
  hid_t datatype_id = H5Dget_type(dataset_id);

  // total number of elements in the buffer
  int64_t total_number_of_elements = 1;
  for (auto i : index_range(dims))
  {
    total_number_of_elements *= dims[i];
  }

  // make tensor
  std::vector<int64_t> torch_dims(dims.begin(), dims.end());

  if (getParam<bool>("is_integer"))
  {
    // create read buffer
    std::vector<int64_t> buffer(total_number_of_elements);
    // read data
    H5Dread(dataset_id, datatype_id, H5S_ALL, dataspace_id, H5P_DEFAULT, buffer.data());

    auto cpu_tensor = torch::from_blob(buffer.data(), torch_dims, torch::kInt64).clone();
    _u = cpu_tensor.to(MooseTensor::intTensorOptions());
  }
  else
  {
    // create read buffer
    std::vector<double> buffer(total_number_of_elements);
    // read data
    H5Dread(dataset_id, datatype_id, H5S_ALL, dataspace_id, H5P_DEFAULT, buffer.data());

    auto cpu_tensor = torch::from_blob(buffer.data(), torch_dims, torch::kFloat64).clone();
    _u = cpu_tensor.to(MooseTensor::floatTensorOptions());
  }
  while (_u.dim() < 3)
    _u.unsqueeze_(-1);

  // close everything
  H5Fclose(file_id);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
#else
  mooseError("MOOSE was built without HDF5 support.");
#endif
}

void
LBMTensorBuffer::makeCPUCopy()
{
  if (!_u.defined())
    return;

  if (_cpu_copy_requested)
  {
    if (_u.is_cpu())
      _u_cpu = _u.clone().contiguous();
    else
      _u_cpu = _u.cpu().contiguous();
  }
}
