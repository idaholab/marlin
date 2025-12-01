/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "XDMFTensorOutput.h"
#include "TensorProblem.h"
#include "Conversion.h"

#include <ATen/core/TensorBody.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <sstream>

#ifdef LIBMESH_HAVE_HDF5
namespace
{
void addDataToHDF5(hid_t file_id,
                   const std::string & dataset_name,
                   const char * data,
                   const std::vector<std::size_t> & ndim,
                   hid_t type);
}
#endif

registerMooseObject("MarlinApp", XDMFTensorOutput);

InputParameters
XDMFTensorOutput::validParams()
{
  auto params = TensorOutput::validParams();
  params.addClassDescription("Output a tensor in XDMF format.");
#ifdef LIBMESH_HAVE_HDF5
  params.addParam<bool>("enable_hdf5", false, "Use HDF5 for binary data storage.");
#endif
  MultiMooseEnum outputMode("CELL NODE OVERSIZED_NODAL");
  outputMode.addDocumentation("CELL", "Output as discontinuous elemental fields.");
  outputMode.addDocumentation(
      "NODE",
      "Output as nodal fields, increasing each box dimension by one and duplicating values from "
      "opposing surfaces to create a periodic continuation.");
  outputMode.addDocumentation("OVERSIZED_NODAL",
                              "Output oversized tensors as nodal fields without forcing "
                              "periodicity (suitable for displacement variables).");

  params.addParam<MultiMooseEnum>("output_mode", outputMode, "Output as cell or node data");
  params.addParam<bool>("transpose",
                        true,
                        "The Paraview XDMF reader swaps x-y (x-z in 3d), so we transpose the "
                        "tensors before we output to make the data look right in Paraview.");
  return params;
}

XDMFTensorOutput::XDMFTensorOutput(const InputParameters & parameters)
  : TensorOutput(parameters),
    _dim(_domain.getDim()),
    _frame(0),
    _rank(_domain.comm().rank()),
    _n_rank(_domain.comm().size()),
    _is_parallel(_n_rank > 1),
    _transpose(getParam<bool>("transpose"))
#ifdef LIBMESH_HAVE_HDF5
    ,
    _enable_hdf5(getParam<bool>("enable_hdf5")),
    _hdf5_name(_file_base + (_is_parallel ? rankTag(_rank) : std::string()) + ".h5")
#endif
{
  const auto output_mode = getParam<MultiMooseEnum>("output_mode").getSetValueIDs<OutputMode>();
  const auto nbuffers = _out_buffers.size();

  if (output_mode.size() == 0)
    // default all to Cell
    for (const auto & pair : _out_buffers)
      _output_mode[pair.first] = OutputMode::CELL;
  else if (output_mode.size() != nbuffers)
    paramError(
        "output_mode", "Specify one output mode per buffer.", output_mode.size(), " != ", nbuffers);
  else
  {
    const auto & buffer_name = getParam<std::vector<TensorInputBufferName>>("buffer");
    for (const auto i : make_range(nbuffers))
      _output_mode[buffer_name[i]] = output_mode[i];
  }

  if (_is_parallel)
    for (const auto & mode_pair : _output_mode)
      if (mode_pair.second != OutputMode::CELL)
        mooseError("XDMFTensorOutput currently supports only CELL output mode in parallel.");

#ifdef LIBMESH_HAVE_HDF5
  // Check if the library is thread-safe
  hbool_t is_threadsafe;
  H5is_library_threadsafe(&is_threadsafe);
  if (!is_threadsafe)
  {
    for (const auto & output : _tensor_problem.getOutputs())
      if (output.get() != this && dynamic_cast<XDMFTensorOutput *>(output.get()))
        mooseError(
            "Using an hdf5 library that is not threadsafe and multiple XDMF output objects. "
            "Consolidate the XDMF outputs or build Marlin with a thread safe build of libhdf5.");
    mooseWarning("Using an hdf5 library that is not threadsafe.");
  }
#endif
}

XDMFTensorOutput::~XDMFTensorOutput()
{
#ifdef LIBMESH_HAVE_HDF5
  if (_enable_hdf5)
    H5Fclose(_hdf5_file_id);
#endif
}

void
XDMFTensorOutput::init()
{
  // get mesh metadata
  auto sdim = Moose::stringify(_dim);
  std::vector<Real> origin;
  std::vector<Real> dgrid;
  for (const auto i : make_range(_dim))
  {
    // we need to transpose the tensor because of
    // https://discourse.paraview.org/t/axis-swapped-with-xdmf-topologytype-3dcorectmesh/3059/4
    const auto j = mappedAxis(i);
    _ndata[0].push_back(_domain.getGridSize()[j]);
    _ndata[1].push_back(_domain.getGridSize()[j] + 1);
    _nnode.push_back(_domain.getGridSize()[j] + 1);
    dgrid.push_back(_domain.getGridSpacing()(j));
    origin.push_back(_domain.getDomainMin()(j));
  }
  _data_grid[0] = Moose::stringify(_ndata[0], " ");
  _data_grid[1] = Moose::stringify(_ndata[1], " ");
  _node_grid = Moose::stringify(_nnode, " ");

  const char * dxyz[] = {"DX", "DY", "DZ"};
  _geometry_type = "ORIGIN_";
  for (const auto i : make_range(_dim))
    _geometry_type += dxyz[i];

  //
  // setup XDMF skeleton
  //

  if (_is_parallel && _rank != 0)
  {
#ifdef LIBMESH_HAVE_HDF5
    if (_enable_hdf5)
    {
      std::filesystem::remove(_hdf5_name);
      _hdf5_file_id = H5Fcreate(_hdf5_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      if (_hdf5_file_id < 0)
      {
        H5Eprint(H5E_DEFAULT, stderr);
        mooseError("Error opening HDF5 file '", _hdf5_name, "'.");
      }
    }
#endif
    return;
  }

  // Top level xdmf block
  auto xdmf = _doc.append_child("Xdmf");
  xdmf.append_attribute("xmlns:xi") = "http://www.w3.org/2003/XInclude";
  xdmf.append_attribute("Version") = "2.2";

  // Domain
  auto domain = xdmf.append_child("Domain");

  // - Topology
  auto topology = domain.append_child("Topology");
  topology.append_attribute("TopologyType") = (sdim + "DCoRectMesh").c_str();
  topology.append_attribute("Dimensions").set_value(_node_grid.c_str());

  // -  Geometry
  auto geometry = domain.append_child("Geometry");
  geometry.append_attribute("Type") = _geometry_type.c_str();

  // -- Origin
  {
    auto data = geometry.append_child("DataItem");
    data.append_attribute("Format").set_value("XML");
    data.append_attribute("Dimensions") = sdim.c_str();
    data.append_child(pugi::node_pcdata).set_value(Moose::stringify(origin, " ").c_str());
  }

  // -- Grid spacing
  {
    auto data = geometry.append_child("DataItem");
    data.append_attribute("Format") = "XML";
    data.append_attribute("Dimensions") = sdim.c_str();
    data.append_child(pugi::node_pcdata).set_value(Moose::stringify(dgrid, " ").c_str());
  }

  // - TimeSeries Grid
  _tgrid = domain.append_child("Grid");
  _tgrid.append_attribute("Name") = "TimeSeries";
  _tgrid.append_attribute("GridType") = "Collection";
  _tgrid.append_attribute("CollectionType") = "Temporal";

  // write XDMF file
  _doc.save_file((_file_base + ".xmf").c_str());
#ifdef LIBMESH_HAVE_HDF5
  // delete HDF5 file
  if (_enable_hdf5)
  {
    std::filesystem::remove(_hdf5_name);
    // open new file
    _hdf5_file_id = H5Fcreate(_hdf5_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (_hdf5_file_id < 0)
    {
      H5Eprint(H5E_DEFAULT, stderr);
      mooseError("Error opening HDF5 file '", _hdf5_name, "'.");
    }
  }
#endif
}

void
XDMFTensorOutput::output()
{
  writeLocalData();

#ifdef LIBMESH_HAVE_HDF5
  if (_enable_hdf5)
    H5Fflush(_hdf5_file_id, H5F_SCOPE_GLOBAL);
#endif

  if (_is_parallel && _rank != 0)
  {
    _frame++;
    return;
  }

  if (_is_parallel)
    writeParallelXMF();
  else
    writeSerialXMF();

  _doc.save_file((_file_base + ".xmf").c_str());

  _frame++;
}

void
XDMFTensorOutput::writeLocalData()
{
  for (const auto & [buffer_name, original_buffer] : _out_buffers)
  {
    auto owned = original_buffer->ownedView();
    if (!owned.defined())
      continue;

    const auto output_mode = _output_mode[buffer_name];
    torch::Tensor buffer;

    switch (output_mode)
    {
      case OutputMode::NODE:
      {
        auto extended = extendTensor(owned);
        buffer = _transpose ? (_dim == 2 ? torch::transpose(extended, 0, 1).contiguous()
                                         : torch::transpose(extended, 0, 2).contiguous())
                            : extended;
        break;
      }

      case OutputMode::OVERSIZED_NODAL:
      case OutputMode::CELL:
      {
        buffer = _transpose ? (_dim == 2 ? torch::transpose(owned, 0, 1).contiguous()
                                         : torch::transpose(owned, 0, 2).contiguous())
                            : owned;
        break;
      }
    }

    const auto sizes = buffer.sizes();
    if (sizes.size() < _dim)
      mooseError("Tensor has fewer dimensions than specified spatial dimension.");

    int64_t num_grid_fields = 1;
    for (const auto i : make_range(_dim))
      num_grid_fields *= sizes[i];

    int64_t num_scalar_fields = 1;
    for (std::size_t i = _dim; i < sizes.size(); ++i)
      num_scalar_fields *= sizes[i];

    std::vector<int64_t> reshape_sizes = {num_grid_fields, num_scalar_fields};
    auto reshaped = buffer.reshape(reshape_sizes);
    auto component_names = buildAttributeNames(buffer_name, num_scalar_fields);

    for (const auto component : make_range(component_names.size()))
    {
      auto slice = reshaped.select(1, component).contiguous();
      if (!slice.device().is_cpu())
        slice = slice.to(slice.options().device(torch::kCPU));

      const auto setname = component_names[component] + "." + Moose::stringify(_frame);
      char * raw_ptr = static_cast<char *>(slice.data_ptr());
      const std::size_t raw_size = slice.nbytes();

#ifdef LIBMESH_HAVE_HDF5
      if (_enable_hdf5)
      {
        std::vector<std::size_t> dims;
        dims.reserve(_dim);
        for (const auto i : make_range(_dim))
          dims.push_back(static_cast<std::size_t>(sizes[i]));

        hid_t hdf_type = H5I_INVALID_HID;
        if (slice.dtype() == torch::kFloat32)
          hdf_type = H5T_NATIVE_FLOAT;
        else if (slice.dtype() == torch::kFloat64)
          hdf_type = H5T_NATIVE_DOUBLE;
        else if (slice.dtype() == torch::kInt32)
          hdf_type = H5T_NATIVE_INT32;
        else if (slice.dtype() == torch::kInt64)
          hdf_type = H5T_NATIVE_INT64;
        else
          mooseError("Unsupported output type");

        addDataToHDF5(_hdf5_file_id, setname, raw_ptr, dims, hdf_type);
      }
      else
#endif
      {
        const auto fname = binaryFileName(setname, _rank);
        auto file = std::fstream(fname.c_str(), std::ios::out | std::ios::binary);
        file.write(raw_ptr, raw_size);
        file.close();
      }
    }
  }
}

void
XDMFTensorOutput::writeSerialXMF()
{
  auto grid = _tgrid.append_child("Grid");
  grid.append_attribute("Name") = ("T" + Moose::stringify(_frame)).c_str();
  grid.append_attribute("GridType") = "Uniform";

  auto time = grid.append_child("Time");
  time.append_attribute("Value") = _time;

  grid.append_child("xi:include").append_attribute("xpointer") = "xpointer(//Xdmf/Domain/Topology)";
  grid.append_child("xi:include").append_attribute("xpointer") = "xpointer(//Xdmf/Domain/Geometry)";

  for (const auto & [buffer_name, original_buffer] : _out_buffers)
  {
    auto owned = original_buffer->ownedView();
    if (!owned.defined())
      continue;

    const auto output_mode = _output_mode[buffer_name];
    const bool is_cell = output_mode == OutputMode::CELL;

    const auto sizes = owned.sizes();
    if (sizes.size() < _dim)
      mooseError("Tensor has fewer dimensions than specified spatial dimension.");

    int64_t num_scalar_fields = 1;
    for (std::size_t i = _dim; i < sizes.size(); ++i)
      num_scalar_fields *= sizes[i];

    const auto component_names = buildAttributeNames(buffer_name, num_scalar_fields);
    const char * center = is_cell ? "Cell" : "Node";
    const auto data_dims = _data_grid[is_cell ? 0 : 1];
    const auto dtype = owned.dtype();

    const char * dtype_str = (dtype == torch::kInt32 || dtype == torch::kInt64) ? "Int" : "Float";
    const std::string precision = (dtype == torch::kFloat64 || dtype == torch::kInt64)   ? "8"
                                  : (dtype == torch::kFloat32 || dtype == torch::kInt32) ? "4"
                                                                                         : "1";

    for (const auto & attr_name : component_names)
    {
      auto attr = grid.append_child("Attribute");
      attr.append_attribute("Name") = attr_name.c_str();
      attr.append_attribute("Center") = center;

      auto data = attr.append_child("DataItem");
      data.append_attribute("DataType") = dtype_str;
      data.append_attribute("Dimensions") = data_dims.c_str();

      const auto dataset = attr_name + "." + Moose::stringify(_frame);

#ifdef LIBMESH_HAVE_HDF5
      if (_enable_hdf5)
      {
        data.append_attribute("Format") = "HDF";
        const auto h5path = _hdf5_name + ":/" + dataset;
        data.append_child(pugi::node_pcdata).set_value(h5path.c_str());
      }
      else
#endif
      {
        data.append_attribute("Format") = "Binary";
        data.append_attribute("Endian") = "Little";
        data.append_attribute("Precision") = precision.c_str();
        const auto fname = binaryFileName(dataset, 0);
        data.append_child(pugi::node_pcdata).set_value(fname.c_str());
      }
    }
  }
}

void
XDMFTensorOutput::writeParallelXMF()
{
  auto grid = _tgrid.append_child("Grid");
  grid.append_attribute("Name") = ("T" + Moose::stringify(_frame)).c_str();
  grid.append_attribute("GridType") = "Collection";
  grid.append_attribute("CollectionType") = "Spatial";

  auto time = grid.append_child("Time");
  time.append_attribute("Value") = _time;

  const auto spacing = localSpacing();
  const std::string spacing_dims = Moose::stringify(_dim);

  for (unsigned int r = 0; r < _n_rank; ++r)
  {
    const auto cells = localCellCounts(r);
    const auto nodes = localNodeCounts(r);
    const auto origin = localOrigin(r);

    auto subgrid = grid.append_child("Grid");
    subgrid.append_attribute("Name") = ("Rank" + Moose::stringify(r)).c_str();
    subgrid.append_attribute("GridType") = "Uniform";

    auto topology = subgrid.append_child("Topology");
    topology.append_attribute("TopologyType") = (Moose::stringify(_dim) + "DCoRectMesh").c_str();
    topology.append_attribute("Dimensions") = dimsToString(nodes).c_str();

    auto geometry = subgrid.append_child("Geometry");
    geometry.append_attribute("Type") = _geometry_type.c_str();

    auto origin_data = geometry.append_child("DataItem");
    origin_data.append_attribute("Format") = "XML";
    origin_data.append_attribute("Dimensions") = spacing_dims.c_str();
    origin_data.append_child(pugi::node_pcdata).set_value(Moose::stringify(origin, " ").c_str());

    auto spacing_data = geometry.append_child("DataItem");
    spacing_data.append_attribute("Format") = "XML";
    spacing_data.append_attribute("Dimensions") = spacing_dims.c_str();
    spacing_data.append_child(pugi::node_pcdata).set_value(Moose::stringify(spacing, " ").c_str());

    for (const auto & [buffer_name, original_buffer] : _out_buffers)
    {
      auto owned = original_buffer->ownedView();
      if (!owned.defined())
        continue;

      const auto sizes = owned.sizes();
      if (sizes.size() < _dim)
        mooseError("Tensor has fewer dimensions than specified spatial dimension.");

      int64_t num_scalar_fields = 1;
      for (std::size_t i = _dim; i < sizes.size(); ++i)
        num_scalar_fields *= sizes[i];

      const auto attr_names = buildAttributeNames(buffer_name, num_scalar_fields);
      const std::string dims_str = dimsToString(cells);
      const auto dtype = owned.dtype();
      const char * dtype_str = (dtype == torch::kInt32 || dtype == torch::kInt64) ? "Int" : "Float";
      const std::string precision = (dtype == torch::kFloat64 || dtype == torch::kInt64)   ? "8"
                                    : (dtype == torch::kFloat32 || dtype == torch::kInt32) ? "4"
                                                                                           : "1";

      for (const auto & attr_name : attr_names)
      {
        auto attr = subgrid.append_child("Attribute");
        attr.append_attribute("Name") = attr_name.c_str();
        attr.append_attribute("Center") = "Cell";

        auto data = attr.append_child("DataItem");
        data.append_attribute("DataType") = dtype_str;
        data.append_attribute("Dimensions") = dims_str.c_str();

        const auto dataset = attr_name + "." + Moose::stringify(_frame);

#ifdef LIBMESH_HAVE_HDF5
        if (_enable_hdf5)
        {
          data.append_attribute("Format") = "HDF";
          const auto h5path = hdf5FileName(r) + ":/" + dataset;
          data.append_child(pugi::node_pcdata).set_value(h5path.c_str());
        }
        else
#endif
        {
          data.append_attribute("Format") = "Binary";
          data.append_attribute("Endian") = "Little";
          data.append_attribute("Precision") = precision.c_str();
          const auto fname = binaryFileName(dataset, r);
          data.append_child(pugi::node_pcdata).set_value(fname.c_str());
        }
      }
    }
  }
}

torch::Tensor
XDMFTensorOutput::extendTensor(torch::Tensor tensor)
{
  // for nodal data we increase each dimension by one and fill in a copy of the slice at 0
  torch::Tensor first;
  using torch::indexing::Slice;

  if (_dim == 3)
  {
    first = tensor.index({0, Slice(), Slice()}).unsqueeze(0);
    tensor = torch::cat({tensor, first}, 0);
    first = tensor.index({Slice(), 0, Slice()}).unsqueeze(1);
    tensor = torch::cat({tensor, first}, 1);
    first = tensor.index({Slice(), Slice(), 0}).unsqueeze(2);
    tensor = torch::cat({tensor, first}, 2);
  }

  else if (_dim == 2)
  {
    first = tensor.index({0}).unsqueeze(0);
    tensor = torch::cat({tensor, first}, 0);
    first = tensor.index({Slice(), 0}).unsqueeze(1);
    tensor = torch::cat({tensor, first}, 1);
  }
  else
    mooseError("Unsupported tensor dimension");

  return tensor.contiguous();
}

torch::Tensor
XDMFTensorOutput::upsampleTensor(torch::Tensor tensor)
{
  // For nodal nonperiodic data we transform into reciprocal space, add one additional K-vector on
  // each dimension, and transform back. This should amount to interpolating the spectral "shape
  // functions" at the nodes, rather than at the cell centers.
  std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
  for (const auto i : make_range(_dim))
    shape[i]++;

  // return back transform with frequency padding
  return torch::fft::irfftn(
      _domain.fft(tensor), torch::IntArrayRef(shape.data(), _dim), _domain.getDimIndices());
}

#ifdef LIBMESH_HAVE_HDF5
namespace
{
void
addDataToHDF5(hid_t file_id,
              const std::string & dataset_name,
              const char * data,
              const std::vector<std::size_t> & ndim,
              hid_t type)
{
  hid_t dataset_id, dataspace_id, plist_id;
  herr_t status;

  // Open the file in read/write mode, create if it doesn't exist

  // hsize_t chunk_dims[RANK];
  std::vector<hsize_t> dims(ndim.begin(), ndim.end());

  // Check if the dataset already exists
  if (H5Lexists(file_id, dataset_name.c_str(), H5P_DEFAULT) > 0)
    mooseError("Dataset '", dataset_name, "' already exists in HDF5 file.");

  // Create a new dataset
  dataspace_id = H5Screate_simple(dims.size(), dims.data(), nullptr);
  if (dataspace_id < 0)
    mooseError("Error creating dataspace");

  plist_id = H5Pcreate(H5P_DATASET_CREATE);
  if (plist_id < 0)
    mooseError("Error creating property list");

  status = H5Pset_chunk(plist_id, dims.size(), dims.data());
  if (status < 0)
    mooseError("Error setting chunking");

  if (H5Zfilter_avail(H5Z_FILTER_DEFLATE))
  {
    unsigned filter_info;
    H5Zget_filter_info(H5Z_FILTER_DEFLATE, &filter_info);
    if (filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED)
    {
      status = H5Pset_deflate(plist_id, 9);
      if (status < 0)
        mooseError("Error setting compression filter");
    }
  }

  dataset_id = H5Dcreate(
      file_id, dataset_name.c_str(), type, dataspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
  if (dataset_id < 0)
  {
    mooseInfo(dataset_id,
              ' ',
              file_id,
              ' ',
              dataset_name.c_str(),
              ' ',
              type,
              ' ',
              dataspace_id,
              ' ',
              H5P_DEFAULT,
              ' ',
              plist_id,
              ' ',
              H5P_DEFAULT);
    mooseError("Error creating dataset");
  }

  // Write data to the dataset
  status = H5Dwrite(dataset_id, type, H5S_ALL, dataspace_id, H5P_DEFAULT, data);

  // Close resources
  H5Pclose(plist_id);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
}
}
#endif
std::vector<std::string>
XDMFTensorOutput::buildAttributeNames(const TensorInputBufferName & buffer_name,
                                      int64_t num_fields) const
{
  const std::array<std::string, 3> xyz = {"x", "y", "z"};
  std::vector<std::string> names;
  names.reserve(num_fields);
  for (const auto index : make_range(static_cast<std::size_t>(num_fields)))
  {
    std::string name = buffer_name;
    if (num_fields > 1)
      name += "_" + (num_fields <= 3 ? xyz[index] : Moose::stringify(index));
    names.push_back(std::move(name));
  }
  return names;
}

unsigned int
XDMFTensorOutput::mappedAxis(unsigned int axis) const
{
  return _transpose ? _dim - axis - 1 : axis;
}

std::vector<int64_t>
XDMFTensorOutput::localCellCounts(unsigned int rank) const
{
  std::array<int64_t, 3> begin = {0, 0, 0};
  std::array<int64_t, 3> end = {0, 0, 0};
  _domain.getLocalBounds(rank, begin, end);

  std::vector<int64_t> counts(_dim, 1);
  for (const auto i : make_range(_dim))
  {
    const auto axis = mappedAxis(i);
    counts[i] = end[axis] - begin[axis];
  }
  return counts;
}

std::vector<int64_t>
XDMFTensorOutput::localNodeCounts(unsigned int rank) const
{
  auto counts = localCellCounts(rank);
  for (auto & c : counts)
    c += 1;
  return counts;
}

std::vector<Real>
XDMFTensorOutput::localOrigin(unsigned int rank) const
{
  std::array<int64_t, 3> begin = {0, 0, 0};
  std::array<int64_t, 3> end = {0, 0, 0};
  _domain.getLocalBounds(rank, begin, end);

  std::vector<Real> origin(_dim, 0.0);
  for (const auto i : make_range(_dim))
  {
    const auto axis = mappedAxis(i);
    origin[i] = _domain.getDomainMin()(axis) + begin[axis] * _domain.getGridSpacing()(axis);
  }
  return origin;
}

std::vector<Real>
XDMFTensorOutput::localSpacing() const
{
  std::vector<Real> spacing(_dim, 0.0);
  for (const auto i : make_range(_dim))
  {
    const auto axis = mappedAxis(i);
    spacing[i] = _domain.getGridSpacing()(axis);
  }
  return spacing;
}

std::string
XDMFTensorOutput::dimsToString(const std::vector<int64_t> & dims) const
{
  return Moose::stringify(dims, " ");
}

std::string
XDMFTensorOutput::rankTag(unsigned int rank) const
{
  if (!_is_parallel)
    return "";
  std::ostringstream os;
  os << ".rank" << std::setfill('0') << std::setw(4) << rank;
  return os.str();
}

std::string
XDMFTensorOutput::hdf5FileName(unsigned int rank) const
{
  return _file_base + rankTag(rank) + ".h5";
}

std::string
XDMFTensorOutput::binaryFileName(const std::string & setname, unsigned int rank) const
{
  return _file_base + rankTag(rank) + "." + setname + ".bin";
}
