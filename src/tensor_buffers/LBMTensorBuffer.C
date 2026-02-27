/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
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
  InputParameters params = PlainTensorBuffer::validParams();
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
  : PlainTensorBuffer(parameters),
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

  std::vector<int64_t> shape = _lb_problem.getLocalTensorShape(std::vector<int64_t>());

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

  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, _domain.comm().get(), MPI_INFO_NULL);
  hid_t file_id = H5Fopen(tensor_file_char, H5F_ACC_RDONLY, plist_id);
  H5Pclose(plist_id);

  if (file_id < 0)
    mooseError("Failed to open h5 file");

  std::string dataset_name = tensor_file_name.substr(0, tensor_file_name.size() - 3);
  auto last_slash = dataset_name.find_last_of("/\\");
  if (last_slash != std::string::npos)
    dataset_name = dataset_name.substr(last_slash + 1);

  hid_t dataset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
  if (dataset_id < 0)
    mooseError("Failed to obtain dataset from h5 file");

  hid_t dataspace_id = H5Dget_space(dataset_id);
  const hsize_t h5_rank = H5Sget_simple_extent_ndims(dataspace_id);
  std::vector<hsize_t> dims(h5_rank);
  H5Sget_simple_extent_dims(dataspace_id, dims.data(), NULL);
  hid_t datatype_id = H5Dget_type(dataset_id);

  auto read_and_process_tensor =
      [&](auto type_dummy, c10::ScalarType torch_dtype, const torch::TensorOptions & moose_options)
  {
    using T = decltype(type_dummy);
    auto r = _domain.comm().rank();
    std::array<int64_t, 3> begin{0, 0, 0}, end{1, 1, 1};

    if (_domain.comm().size() > 1)
      _domain.getLocalBounds(r, begin, end);
    else
      for (int i = 0; i < static_cast<int>(h5_rank); i++)
      {
        begin[i] = 0;
        end[i] = dims[i];
      }

    std::vector<hsize_t> start(h5_rank), count(h5_rank);
    int64_t local_elements = 1;
    for (int i = 0; i < static_cast<int>(h5_rank); i++)
    {
      start[i] = begin[i];
      count[i] = end[i] - begin[i];
      local_elements *= count[i];
    }

    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, start.data(), NULL, count.data(), NULL);
    hid_t memspace_id = H5Screate_simple(h5_rank, count.data(), NULL);

    hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);

    std::vector<T> buffer(local_elements);
    H5Dread(dataset_id, datatype_id, memspace_id, dataspace_id, xfer_plist, buffer.data());

    std::vector<int64_t> local_torch_dims(count.begin(), count.end());
    auto local_cpu_tensor = torch::from_blob(buffer.data(), local_torch_dims, torch_dtype).clone();

    _u = torch::ones(_lb_problem.getLocalTensorShape(std::vector<int64_t>()), moose_options);
    auto ghost_radius = _lb_problem.getGhostRadius();
    torch::Tensor u_owned = _u;
    for (unsigned int d = 0; d < _domain.getDim(); d++)
      u_owned = u_owned.narrow(d, ghost_radius, local_cpu_tensor.size(d));

    u_owned.copy_(local_cpu_tensor.to(moose_options));

    H5Pclose(xfer_plist);
    H5Sclose(memspace_id);
  };

  if (getParam<bool>("is_integer"))
    read_and_process_tensor(int64_t{}, torch::kInt64, MooseTensor::intTensorOptions());
  else
    read_and_process_tensor(double{}, torch::kFloat64, MooseTensor::floatTensorOptions());

  while (_u.dim() < 3)
    _u.unsqueeze_(-1);

  H5Tclose(datatype_id);
  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);
#else
  mooseError("MOOSE was built without HDF5 support.");
#endif
}
