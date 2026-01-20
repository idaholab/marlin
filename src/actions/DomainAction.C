/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "DomainAction.h"
#include "MooseError.h"
#include "TensorProblem.h"
#include "MooseEnum.h"
#include "MultiMooseEnum.h"
#include "MooseUtils.h"
#include "SetupMeshAction.h"
#include "MarlinApp.h"
#include "CreateProblemAction.h"

#include <initializer_list>
#include <util/Optional.h>
#include <cmath>
#include <limits>

// run this early, before any objects are constructed
registerMooseAction("MarlinApp", DomainAction, "meta_action");
registerMooseAction("MarlinApp", DomainAction, "add_mesh_generator");
registerMooseAction("MarlinApp", DomainAction, "create_problem_custom");

InputParameters
DomainAction::validParams()
{
  InputParameters params = Action::validParams();
  params.addClassDescription("Set up the domain and compute devices.");

  MooseEnum dims("1=1 2 3");
  params.addRequiredParam<MooseEnum>("dim", dims, "Problem dimension");

  MooseEnum parmode("NONE REAL_SPACE FFT_SLAB FFT_PENCIL", "NONE");
  parmode.addDocumentation("NONE", "Serial execution without domain decomposition.");
  parmode.addDocumentation("REAL_SPACE", "Real-space domain decomposition with halo exchanges.");
  parmode.addDocumentation("FFT_SLAB",
                           "Slab decomposition with X-Z slabs stacked along the Y direction in "
                           "real space and Y-Z slabs stacked along the X direction in Fourier "
                           "space. This requires one all-to-all communication per FFT.");
  parmode.addDocumentation(
      "FFT_PENCIL",
      "Pencil decomposition (3D only). Three 1D FFTs in pencil arrays along the X, Y, and lastly Z "
      "direction. Thie requires two many-to-many communications per FFT.");

  params.addParam<MooseEnum>("parallel_mode", parmode, "Parallelization mode.");
  MultiMooseEnum periodic_enum("X=0 Y=1 Z=2");
  params.addParam<MultiMooseEnum>(
      "periodic_directions",
      periodic_enum,
      "Periodic directions of the simulation cell (controls halo exchange wrap-around).");

  params.addParam<unsigned int>("nx", 1, "Number of elements in the X direction");
  params.addParam<unsigned int>("ny", 1, "Number of elements in the Y direction");
  params.addParam<unsigned int>("nz", 1, "Number of elements in the Z direction");
  params.addParam<Real>("xmax", 1.0, "Upper X Coordinate of the generated mesh");
  params.addParam<Real>("ymax", 1.0, "Upper Y Coordinate of the generated mesh");
  params.addParam<Real>("zmax", 1.0, "Upper Z Coordinate of the generated mesh");
  params.addParam<Real>("xmin", 0.0, "Lower X Coordinate of the generated mesh");
  params.addParam<Real>("ymin", 0.0, "Lower Y Coordinate of the generated mesh");
  params.addParam<Real>("zmin", 0.0, "Lower Z Coordinate of the generated mesh");

  MooseEnum meshmode("DUMMY DOMAIN MANUAL", "DUMMY");
  meshmode.addDocumentation("DUMMY",
                            "Create a single element mesh the size of the simulation domain");
  meshmode.addDocumentation("DOMAIN", "Create a mesh with one element per grid cell");
  meshmode.addDocumentation("MANUAL",
                            "Do not auto-generate a mesh. User must add a Mesh block themselves.");

  params.addParam<MooseEnum>("mesh_mode", meshmode, "Mesh generation mode.");

  params.addParam<std::vector<std::string>>("device_names", {}, "Compute devices to run on.");
  params.addParam<std::vector<unsigned int>>(
      "device_weights", {}, "Device weights (or speeds) to influence the partitioning.");

  MooseEnum floatingPrecision("DEVICE_DEFAULT SINGLE DOUBLE", "DEVICE_DEFAULT");
  params.addParam<MooseEnum>("floating_precision", floatingPrecision, "Floating point precision.");

  params.addParam<bool>(
      "debug",
      false,
      "Enable additional debugging and diagnostics, such a checking for initialized tensors.");
  params.addParam<bool>("gpu_aware_mpi",
                        false,
                        "Enable GPU-aware MPI. If true, tensors will not be copied to the CPU "
                        "before MPI communication. Requires a CUDA-aware MPI implementation.");
  return params;
}

DomainAction::DomainAction(const InputParameters & parameters)
  : Action(parameters),
    _device_names(getParam<std::vector<std::string>>("device_names")),
    _device_weights(getParam<std::vector<unsigned int>>("device_weights")),
    _floating_precision(getParam<MooseEnum>("floating_precision").getEnum<FloatingPrecision>()),
    _parallel_mode(getParam<MooseEnum>("parallel_mode").getEnum<ParallelMode>()),
    _periodic(
        [&]()
        {
          std::array<bool, 3> p{{false, false, false}};
          const auto & periodic_dirs = getParam<MultiMooseEnum>("periodic_directions");
          for (unsigned int i = 0; i < periodic_dirs.size(); ++i)
          {
            const auto id = periodic_dirs.get(i);
            if (id < 3)
              p[id] = true;
          }
          return p;
        }()),
    _dim(getParam<MooseEnum>("dim")),
    _n_global(
        {getParam<unsigned int>("nx"), getParam<unsigned int>("ny"), getParam<unsigned int>("nz")}),
    _min_global({getParam<Real>("xmin"), getParam<Real>("ymin"), getParam<Real>("zmin")}),
    _max_global({getParam<Real>("xmax"), getParam<Real>("ymax"), getParam<Real>("zmax")}),
    _mesh_mode(getParam<MooseEnum>("mesh_mode").getEnum<MeshMode>()),
    _shape(torch::IntArrayRef(_n_local.data(), _dim)),
    _reciprocal_shape(torch::IntArrayRef(_n_reciprocal_local.data(), _dim)),
    _domain_dimensions_buffer({0, 1, 2}),
    _domain_dimensions(torch::IntArrayRef(_domain_dimensions_buffer.data(), _dim)),
    _rank(_communicator.rank()),
    _n_rank(_communicator.size()),
    _send_tensor(_n_rank),
    _recv_tensor(_n_rank),
    _debug(getParam<bool>("debug")),
    _gpu_aware_mpi(getParam<bool>("gpu_aware_mpi"))
{
  for (const auto d : make_range(3u))
  {
    _local_begin[d].assign(_n_rank, 0);
    _local_end[d].assign(_n_rank, 0);
    _n_local_all[d].assign(_n_rank, 0);
  }

  if (_parallel_mode == ParallelMode::NONE && _n_rank > 1)
    paramError("parallel_mode", "NONE requires the application to run in serial.");

  // TODO: Implement non-periodic BCs
  if (_parallel_mode == ParallelMode::REAL_SPACE)
    for (const auto i : make_range(_dim))
      if (!_periodic[i])
        paramError("periodic_directions",
                   "Domain must be periodic in all directions with `parallel_mode = REAL_SPACE`.");

  auto marlin_app = dynamic_cast<MarlinApp *>(&_app);
  if (_n_rank == 1)
  {
    // set local weights and ranks for serial
    _local_ranks = {0};
    _local_weights = {1};

    if (_device_names.size() && !marlin_app->parameters().isParamSetByUser("compute_device"))
      marlin_app->setTorchDevice(_device_names[0], {});
  }
  else
  {
    // process weights
    if (_device_weights.empty())
      _device_weights.assign(_n_rank, 1);

    // determine the processor name
    char name[MPI_MAX_PROCESSOR_NAME + 1];
    int len;
    MPI_Get_processor_name(name, &len);
    name[len] = 0;

    // gather all processor names
    std::vector<std::string> host_names;
    _communicator.allgather(std::string(name), host_names);

    // get the local rank on the current processor (used for compute device assignment)
    std::map<std::string, unsigned int> host_rank_count;

    for (const auto & host_name : host_names)
    {
      if (host_rank_count.find(host_name) == host_rank_count.end())
        host_rank_count[host_name] = 0;

      auto & local_rank = host_rank_count[host_name];
      _local_ranks.push_back(local_rank);
      _local_weights.push_back(_device_weights[local_rank % _device_weights.size()]);

      // std::cout << "Process on " << host_name << ' ' << local_rank << ' '
      //           << _device_weights[local_rank % _device_weights.size()] << '\n';

      local_rank++;
    }

    // for (const auto i : index_range(host_names))
    //   std::cout << host_names[i] << '\t' << _local_ranks[i] << '\n';

    // pick a compute device for a list of available devices
    if (!marlin_app)
      mooseError("This action requires a MarlinApp object to be present.");
    if (_device_names.size())
      marlin_app->setTorchDevice(_device_names[_local_ranks[_rank] % _device_names.size()], {});
  }

  switch (_floating_precision)
  {
    case FloatingPrecision::DEVICE_DEFAULT:
    {
      marlin_app->setTorchPrecision("DEVICE_DEFAULT", {});
      break;
    }
    case FloatingPrecision::DOUBLE:
    {
      marlin_app->setTorchPrecision("DOUBLE", {});
      break;
    }
    case FloatingPrecision::SINGLE:
    {
      marlin_app->setTorchPrecision("SINGLE", {});
      break;
    }
    default:
      mooseError("Invalid floating precision.");
  };

  // domain partitioning
  gridChanged();
}

void
DomainAction::gridChanged()
{
  auto options = MooseTensor::floatTensorOptions();
  const bool real_space = _parallel_mode == ParallelMode::REAL_SPACE;

  // build real space axes
  _volume_global = 1.0;
  for (const unsigned int dim : {0, 1, 2})
  {
    // error check
    if (_max_global(dim) <= _min_global(dim))
      mooseError("Max coordinate must be larger than the min coordinate in every dimension");

    // get grid geometry
    _grid_spacing(dim) = (_max_global(dim) - _min_global(dim)) / _n_global[dim];

    // real space axis
    if (dim < _dim)
    {
      _global_axis[dim] =
          align(torch::linspace(c10::Scalar(_min_global(dim) + _grid_spacing(dim) / 2.0),
                                c10::Scalar(_max_global(dim) - _grid_spacing(dim) / 2.0),
                                _n_global[dim],
                                options),
                dim);
      _volume_global *= _max_global(dim) - _min_global(dim);
    }
    else
      _global_axis[dim] = torch::tensor({0.0}, options);
  }

  // build reciprocal space axes
  for (const unsigned int dim : {0, 1, 2})
  {
    if (real_space)
    {
      _global_reciprocal_axis[dim] = torch::tensor({}, options);
      _n_reciprocal_global[dim] = 0;
      continue;
    }

    if (dim < _dim)
    {
      bool use_rfft = false;
      switch (_parallel_mode)
      {
        case ParallelMode::NONE:
          use_rfft = (dim == _dim - 1);
          break;
        case ParallelMode::REAL_SPACE:
          use_rfft = false;
          break;
        case ParallelMode::FFT_SLAB:
          use_rfft = false;
          break;
        case ParallelMode::FFT_PENCIL:
          use_rfft = (dim == 0);
          break;
      }
      const auto freq = use_rfft ? torch::fft::rfftfreq(_n_global[dim], _grid_spacing(dim), options)
                                 : torch::fft::fftfreq(_n_global[dim], _grid_spacing(dim), options);

      // zero out nyquist frequency
      // if (_n_global[dim] % 2 == 0)
      //   freq[_n_global[dim] / 2] = 0.0;

      _global_reciprocal_axis[dim] = align(freq * 2.0 * libMesh::pi, dim);
    }
    else
      _global_reciprocal_axis[dim] = torch::tensor({0.0}, options);

    // compute max frequency along each axis
    _max_k(dim) = libMesh::pi / _grid_spacing(dim);

    // get global reciprocal axis size
    _n_reciprocal_global[dim] = _global_reciprocal_axis[dim].sizes()[dim];
  }

  switch (_parallel_mode)
  {
    case ParallelMode::NONE:
      partitionSerial();
      break;

    case ParallelMode::REAL_SPACE:
      partitionRealSpace();
      break;

    case ParallelMode::FFT_SLAB:
      partitionSlabs();
      break;

    case ParallelMode::FFT_PENCIL:
      partitionPencils();
      break;
  }

  // get local reciprocal axis size
  if (!real_space)
    for (const auto dim : {0, 1, 2})
      _n_reciprocal_local[dim] = _local_reciprocal_axis[dim].sizes()[dim];
  else
    _n_reciprocal_local = {0, 0, 0};

  // update on-demand grids
  if (_x_grid.defined())
    updateXGrid();
  if (_k_grid.defined())
    updateKGrid();
  if (_k_square.defined())
    updateKSquare();
}

void
DomainAction::partitionSerial()
{
  // goes along the full dimension for each rank
  for (const auto d : make_range(3u))
  {
    _local_begin[d].assign(_n_rank, 0);
    _local_end[d].assign(_n_rank, _n_global[d]);
    _n_local_all[d].assign(_n_rank, _n_global[d]);
  }

  // to do, make those slices dependent on local begin/end
  _local_axis = _global_axis;
  _n_local = _n_global;
  _local_reciprocal_axis = _global_reciprocal_axis;
}

void
DomainAction::partitionRealSpace()
{
  // determine factorization (near cubic surface area minimization)
  const auto length = [&](unsigned int d) { return (_max_global(d) - _min_global(d)); };
  const double lx = length(0);
  const double ly = (_dim > 1) ? length(1) : 1.0;
  const double lz = (_dim > 2) ? length(2) : 1.0;

  auto cost2d = [&](unsigned int nx, unsigned int ny)
  {
    const double hx = lx / nx;
    const double hy = ly / ny;
    return 2.0 * (hx + hy);
  };
  auto cost3d = [&](unsigned int nx, unsigned int ny, unsigned int nz)
  {
    const double hx = lx / nx;
    const double hy = ly / ny;
    const double hz = lz / nz;
    return 2.0 * (hy * hz + hx * hz + hx * hy);
  };

  std::array<unsigned int, 3> best{{1, 1, 1}};
  double best_cost = std::numeric_limits<double>::max();
  double best_aspect = std::numeric_limits<double>::max();

  if (_dim == 2)
  {
    for (unsigned int nx = 1; nx <= _n_rank; ++nx)
      if (_n_rank % nx == 0)
      {
        const unsigned int ny = _n_rank / nx;
        if (nx > _n_global[0] || ny > _n_global[1])
          continue;
        const double c = cost2d(nx, ny);
        const double aspect = std::abs((lx / nx) - (ly / ny));
        if (c < best_cost || (MooseUtils::absoluteFuzzyEqual(c, best_cost) && aspect < best_aspect))
        {
          best_cost = c;
          best_aspect = aspect;
          best = {nx, ny, 1};
        }
      }
  }
  else if (_dim == 3)
  {
    for (unsigned int nx = 1; nx <= _n_rank; ++nx)
      if (_n_rank % nx == 0)
      {
        const unsigned int rem = _n_rank / nx;
        for (unsigned int ny = 1; ny <= rem; ++ny)
          if (rem % ny == 0)
          {
            const unsigned int nz = rem / ny;
            if (nx > _n_global[0] || ny > _n_global[1] || nz > _n_global[2])
              continue;
            const double c = cost3d(nx, ny, nz);
            const double max_len = std::max({lx / nx, ly / ny, lz / nz});
            const double min_len = std::min({lx / nx, ly / ny, lz / nz});
            const double aspect = max_len - min_len;
            if (c < best_cost ||
                (MooseUtils::absoluteFuzzyEqual(c, best_cost) && aspect < best_aspect))
            {
              best_cost = c;
              best_aspect = aspect;
              best = {nx, ny, nz};
            }
          }
      }
  }
  else
    mooseError("Unsupported dimension.");

  if (best_cost == std::numeric_limits<double>::max())
    mooseError("Unable to factor ", _n_rank, " ranks into a near-cubic real-space grid that fits.");

  _real_space_partitions = best;

  auto buildOffsets = [](const std::vector<int64_t> & counts)
  {
    std::vector<int64_t> offsets(counts.size(), 0);
    int64_t cursor = 0;
    for (const auto i : index_range(counts))
    {
      offsets[i] = cursor;
      cursor += counts[i];
    }
    return offsets;
  };

  std::array<std::vector<int64_t>, 3> counts;
  std::array<std::vector<int64_t>, 3> offsets;

  counts[0] =
      partitionHepler<int64_t>(_n_global[0], std::vector<int64_t>(_real_space_partitions[0], 1));
  counts[1] =
      partitionHepler<int64_t>(_n_global[1], std::vector<int64_t>(_real_space_partitions[1], 1));
  counts[2] =
      partitionHepler<int64_t>(_n_global[2], std::vector<int64_t>(_real_space_partitions[2], 1));

  offsets[0] = buildOffsets(counts[0]);
  offsets[1] = buildOffsets(counts[1]);
  offsets[2] = buildOffsets(counts[2]);

  for (const auto d : {0u, 1u, 2u})
    _n_local_all[d].assign(_n_rank, 0);

  for (unsigned int r = 0; r < _n_rank; ++r)
  {
    const unsigned int ix = r % _real_space_partitions[0];
    const unsigned int iy = (r / _real_space_partitions[0]) % _real_space_partitions[1];
    const unsigned int iz = r / (_real_space_partitions[0] * _real_space_partitions[1]);

    _local_begin[0][r] = offsets[0][ix];
    _local_end[0][r] = offsets[0][ix] + counts[0][ix];
    _n_local_all[0][r] = counts[0][ix];

    _local_begin[1][r] = offsets[1][iy];
    _local_end[1][r] = offsets[1][iy] + counts[1][iy];
    _n_local_all[1][r] = counts[1][iy];

    _local_begin[2][r] = offsets[2][iz];
    _local_end[2][r] = offsets[2][iz] + counts[2][iz];
    _n_local_all[2][r] = counts[2][iz];
  }

  _real_space_index = {
      static_cast<unsigned int>(_rank % _real_space_partitions[0]),
      static_cast<unsigned int>((_rank / _real_space_partitions[0]) % _real_space_partitions[1]),
      static_cast<unsigned int>(_rank / (_real_space_partitions[0] * _real_space_partitions[1]))};

  _n_local[0] = counts[0][_real_space_index[0]];
  _n_local[1] = counts[1][_real_space_index[1]];
  _n_local[2] = counts[2][_real_space_index[2]];

  _local_axis[0] = _global_axis[0].slice(0, _local_begin[0][_rank], _local_end[0][_rank]);
  if (_dim > 1)
    _local_axis[1] = _global_axis[1].slice(1, _local_begin[1][_rank], _local_end[1][_rank]);
  else
    _local_axis[1] = _global_axis[1];

  if (_dim > 2)
    _local_axis[2] = _global_axis[2].slice(2, _local_begin[2][_rank], _local_end[2][_rank]);
  else
    _local_axis[2] = _global_axis[2];

  // no reciprocal space decomposition in real-space mode
  const auto reciprocal_options = MooseTensor::complexFloatTensorOptions();
  for (const auto d : {0u, 1u, 2u})
    _local_reciprocal_axis[d] = torch::tensor({}, reciprocal_options);
}

void
DomainAction::partitionSlabs()
{
  if (_dim < 2)
    paramError("dim", "Dimension must be 2 or 3 for slab decomposition.");

  if (_local_weights.size() != _n_rank)
    mooseError("Internal error: local weight vector size does not match number of ranks.");

  // x is partitioned along the reciprocal axis (rfft halves the dimension)
  _n_local_all[0] = partitionHepler(_global_reciprocal_axis[0].sizes()[0], _local_weights);

  // y is partitioned along the real-space axis
  _n_local_all[1] = partitionHepler(_global_axis[1].sizes()[1], _local_weights);

  // set begin/end for x and y
  for (const auto d : {0, 1})
  {
    int64_t b = 0;
    for (const auto r : make_range(_n_rank))
    {
      _local_begin[d][r] = b;
      b += _n_local_all[d][r];
      _local_end[d][r] = b;
    }
  }

  // z is not partitioned at all
  _n_local_all[2].assign(_n_rank, _n_global[2]);
  _local_begin[2].assign(_n_rank, 0);
  _local_end[2].assign(_n_rank, _n_global[2]);

  // slice the real space into x-z slabs stacked in y direction
  _local_axis[0] = _global_axis[0].slice(0, 0, _n_global[0]);
  _local_axis[1] = _global_axis[1].slice(1, _local_begin[1][_rank], _local_end[1][_rank]);
  _n_local[0] = _n_global[0];
  _n_local[1] = _local_end[1][_rank] - _local_begin[1][_rank];

  // slice the reciprocal space into y-z slices stacked in x direction
  _local_reciprocal_axis[0] =
      _global_reciprocal_axis[0].slice(0, _local_begin[0][_rank], _local_end[0][_rank]);
  _local_reciprocal_axis[1] = _global_reciprocal_axis[1].slice(1, 0, _n_reciprocal_global[1]);

  _n_local[2] = _n_global[2];

  // special casing this should not be neccessary
  if (_dim == 3)
  {
    _local_axis[2] = _global_axis[2].slice(2, 0, _n_global[2]);
    _local_reciprocal_axis[2] = _global_reciprocal_axis[2].slice(2, 0, _n_reciprocal_global[2]);
  }
  else
  {
    _local_axis[2] = _global_axis[2];
    _local_reciprocal_axis[2] = _global_reciprocal_axis[2];
  }
}

void
DomainAction::partitionPencils()
{
  if (_dim < 3)
    paramError("dim", "Dimension must be 3 for pencil decomposition.");

  const auto canUseFactors = [&](unsigned int px, unsigned int pz)
  {
    if (px < 2 || pz < 2)
      return false;
    if (px > _n_global[1] || px > _n_reciprocal_global[0])
      return false;
    if (pz > _n_global[2] || pz > _n_global[1])
      return false;
    return true;
  };

  std::pair<unsigned int, unsigned int> best = {0, 0};
  bool found = false;
  unsigned int best_cost = std::numeric_limits<unsigned int>::max();

  auto consider = [&](unsigned int px, unsigned int pz)
  {
    if (!canUseFactors(px, pz))
      return;
    const unsigned int cost = std::abs(static_cast<int>(px) - static_cast<int>(pz));
    if (!found || cost < best_cost)
    {
      best = {px, pz};
      best_cost = cost;
      found = true;
    }
  };

  const unsigned int max_divisor = std::max(2u, static_cast<unsigned int>(std::sqrt(_n_rank)));
  for (unsigned int d = 2; d <= max_divisor; ++d)
    if (_n_rank % d == 0)
    {
      const unsigned int other = _n_rank / d;
      consider(d, other);
      consider(other, d);
    }

  if (!found)
    paramError("parallel_mode",
               "FFT_PENCIL requires factoring the number of MPI ranks into two integers greater "
               "than one that fit the domain (ranks = ",
               _n_rank,
               "). Use FFT_SLAB or adjust the rank count.");

  _pencil_y_partitions = best.first;
  _pencil_z_partitions = best.second;

  auto buildOffsets = [](const std::vector<int64_t> & counts)
  {
    std::vector<int64_t> offsets(counts.size(), 0);
    int64_t cursor = 0;
    for (const auto i : index_range(counts))
    {
      offsets[i] = cursor;
      cursor += counts[i];
    }
    return offsets;
  };

  std::vector<int64_t> unit_y_weights(_pencil_y_partitions, 1);
  std::vector<int64_t> unit_z_weights(_pencil_z_partitions, 1);

  auto y_counts = partitionHepler<int64_t>(_n_global[1], unit_y_weights);
  auto z_counts = partitionHepler<int64_t>(_n_global[2], unit_z_weights);
  auto y_offsets = buildOffsets(y_counts);
  auto z_offsets = buildOffsets(z_counts);

  _pencil_y_index.resize(_n_rank);
  _pencil_z_index.resize(_n_rank);

  for (unsigned int r = 0; r < _n_rank; ++r)
  {
    const unsigned int py = r % _pencil_y_partitions;
    const unsigned int pz = r / _pencil_y_partitions;
    _pencil_y_index[r] = py;
    _pencil_z_index[r] = pz;

    _local_begin[0][r] = 0;
    _local_end[0][r] = _n_global[0];
    _n_local_all[0][r] = _n_global[0];

    _local_begin[1][r] = y_offsets[py];
    _local_end[1][r] = y_offsets[py] + y_counts[py];
    _n_local_all[1][r] = y_counts[py];

    _local_begin[2][r] = z_offsets[pz];
    _local_end[2][r] = z_offsets[pz] + z_counts[pz];
    _n_local_all[2][r] = z_counts[pz];
  }

  _n_local[0] = _n_global[0];
  _n_local[1] = _n_local_all[1][_rank];
  _n_local[2] = _n_local_all[2][_rank];

  _local_axis[0] = _global_axis[0];
  _local_axis[1] = _global_axis[1].slice(1, _local_begin[1][_rank], _local_end[1][_rank]);
  _local_axis[2] = _global_axis[2].slice(2, _local_begin[2][_rank], _local_end[2][_rank]);

  // reciprocal partitions
  _pencil_x_offsets.clear();
  _pencil_x_sizes.clear();
  _pencil_x_offsets.resize(_pencil_y_partitions, 0);
  _pencil_x_sizes = partitionHepler<int64_t>(_n_reciprocal_global[0],
                                             std::vector<int64_t>(_pencil_y_partitions, 1));
  int64_t cursor = 0;
  for (const auto i : index_range(_pencil_x_sizes))
  {
    _pencil_x_offsets[i] = cursor;
    cursor += _pencil_x_sizes[i];
  }

  _pencil_stage2_y_sizes = partitionHepler<int64_t>(_n_reciprocal_global[1],
                                                    std::vector<int64_t>(_pencil_z_partitions, 1));
  _pencil_stage2_y_offsets = buildOffsets(_pencil_stage2_y_sizes);

  const unsigned int px = _rank % _pencil_y_partitions;
  const unsigned int py_final = _rank / _pencil_y_partitions;

  _local_reciprocal_axis[0] = _global_reciprocal_axis[0].slice(
      0, _pencil_x_offsets[px], _pencil_x_offsets[px] + _pencil_x_sizes[px]);
  _local_reciprocal_axis[1] = _global_reciprocal_axis[1].slice(
      1,
      _pencil_stage2_y_offsets[py_final],
      _pencil_stage2_y_offsets[py_final] + _pencil_stage2_y_sizes[py_final]);
  _local_reciprocal_axis[2] = _global_reciprocal_axis[2];

  const auto local_kx = _pencil_x_sizes[_pencil_y_index[_rank]];
  const auto local_ky = _pencil_stage2_y_sizes[_pencil_z_index[_rank]];
  const auto local_kz = _n_reciprocal_global[2];

  if (_rank == 0)
    mooseInfo("FFT_PENCIL decomposition: ",
              _n_rank,
              " ranks = ",
              _pencil_y_partitions,
              " x-pencils × ",
              _pencil_z_partitions,
              " z-slabs (real ",
              "local=",
              _n_local[0],
              "×",
              _n_local[1],
              "×",
              _n_local[2],
              ", reciprocal local=",
              local_kx,
              "×",
              local_ky,
              "×",
              local_kz,
              ")");

  if (_debug)
    mooseInfo("Rank ",
              _n_rank,
              " pencil layout -> real [",
              _n_local_all[0][_rank],
              "×",
              _n_local_all[1][_rank],
              "×",
              _n_local_all[2][_rank],
              ", reciprocal [",
              local_kx,
              "×",
              local_ky,
              "×",
              local_kz,
              "]");
}

void
DomainAction::act()
{
  if (_current_task == "meta_action" && _mesh_mode != MeshMode::MARLIN_MANUAL)
  {
    // check if a SetupMesh action exists
    auto mesh_actions = _awh.getActions<SetupMeshAction>();
    if (mesh_actions.size() > 0)
      paramError("mesh_mode", "Do not specify a [Mesh] block unless mesh_mode is set to MANUAL");

    // otherwise create one
    auto & af = _app.getActionFactory();
    InputParameters action_params = af.getValidParams("SetupMeshAction");
    auto action = std::static_pointer_cast<MooseObjectAction>(
        af.create("SetupMeshAction", "Mesh", action_params));
    _app.actionWarehouse().addActionBlock(action);
  }

  // add a DomainMeshGenerator
  if (_current_task == "add_mesh_generator" && _mesh_mode != MeshMode::MARLIN_MANUAL)
  {
    // Don't do mesh generators when recovering or when the user has requested for us not to
    if ((_app.isRecovering() && _app.isUltimateMaster()) || _app.masterMesh())
      return;

    const MeshGeneratorName name = "domain_mesh_generator";
    auto params = _factory.getValidParams("DomainMeshGenerator");

    params.set<MooseEnum>("dim") = _dim;
    params.set<Real>("xmax") = _max_global(0);
    params.set<Real>("ymax") = _max_global(1);
    params.set<Real>("zmax") = _max_global(2);
    params.set<Real>("xmin") = _min_global(0);
    params.set<Real>("ymin") = _min_global(1);
    params.set<Real>("zmin") = _min_global(2);

    if (_mesh_mode == MeshMode::MARLIN_DOMAIN)
    {
      params.set<unsigned int>("nx") = _n_global[0];
      params.set<unsigned int>("ny") = _n_global[1];
      params.set<unsigned int>("nz") = _n_global[2];
    }
    else if (_mesh_mode == MeshMode::MARLIN_DUMMY)
    {
      params.set<unsigned int>("nx") = 1;
      params.set<unsigned int>("ny") = 1;
      params.set<unsigned int>("nz") = 1;
    }
    else
      mooseError("Internal error");

    _app.addMeshGenerator("DomainMeshGenerator", name, params);
  }

  if (_current_task == "create_problem_custom")
  {
    if (!_problem)
    {
      const std::string type = "TensorProblem";
      auto params = _factory.getValidParams(type);

      // apply common parameters of the object held by CreateProblemAction
      // to honor user inputs in [Problem]
      auto p = _awh.getActionByTask<CreateProblemAction>("create_problem");
      if (p)
        params.applyParameters(p->getObjectParams());

      params.set<MooseMesh *>("mesh") = _mesh.get();
      _problem = _factory.create<FEProblemBase>(type, "MOOSE Problem", params);
    }
  }
}

const torch::Tensor &
DomainAction::getAxis(std::size_t component) const
{
  if (component < 3)
    return _local_axis[component];
  mooseError("Invalid component");
}

const torch::Tensor &
DomainAction::getReciprocalAxis(std::size_t component) const
{
  if (component < 3)
    return _local_reciprocal_axis[component];
  mooseError("Invalid component");
}

torch::Tensor
DomainAction::fft(const torch::Tensor & t) const
{
  switch (_parallel_mode)
  {
    case ParallelMode::NONE:
      return fftSerial(t);

    case ParallelMode::REAL_SPACE:
      mooseError("FFT is not available in REAL_SPACE parallel mode.");

    case ParallelMode::FFT_SLAB:
      return fftSlab(t);

    case ParallelMode::FFT_PENCIL:
      return fftPencil(t);
  }
  mooseError("Not implemented");
}

torch::Tensor
DomainAction::fftSerial(const torch::Tensor & t) const
{
  switch (_dim)
  {
    case 1:
      return torch::fft::rfft(t, c10::nullopt, 0);
    case 2:
      return torch::fft::rfft2(t, c10::nullopt, {0, 1});
    case 3:
      return torch::fft::rfftn(t, c10::nullopt, {0, 1, 2});
    default:
      mooseError("Unsupported mesh dimension");
  }
}

torch::Tensor
DomainAction::fftSlab(const torch::Tensor & t) const
{
  mooseInfoRepeated("fftSlab");
  if (_dim == 1)
    mooseError("Unsupported mesh dimension");

  MooseTensor::printTensorInfo(t);

  auto slab =
      _dim == 3 ? torch::fft::fft2(t, c10::nullopt, {0, 2}) : torch::fft::fft(t, c10::nullopt, 0);
  MooseTensor::printTensorInfo(slab);

  const auto mpi_type = mpiTypeFromScalar(slab.scalar_type());
  const auto cpu_options = slab.options().device(torch::kCPU);
  const auto device_options = slab.options();

  std::vector<MPI_Request> send_requests(_n_rank, MPI_REQUEST_NULL);
  for (const auto i : make_range(_n_rank))
  {
    auto slice = slab.slice(0, _local_begin[0][i], _local_end[0][i]).contiguous();
    if (i == _rank)
      _recv_tensor[i] = slice;
    else
    {
      if (_gpu_aware_mpi)
        _send_tensor[i] = slice;
      else
        _send_tensor[i] = slice.to(cpu_options);

      MPI_Isend(_send_tensor[i].data_ptr(),
                _send_tensor[i].numel(),
                mpi_type,
                i,
                0,
                mpiComm(),
                &send_requests[i]);
    }
  }

  for (const auto i : make_range(_n_rank))
    if (i != _rank)
    {
      std::vector<int64_t> recv_shape;
      if (_dim == 2)
        recv_shape = {_n_local_all[0][_rank], _n_local_all[1][i]};
      else
        recv_shape = {_n_local_all[0][_rank], _n_local_all[1][i], _n_local_all[2][i]};

      auto recv_tensor = torch::empty(recv_shape, _gpu_aware_mpi ? device_options : cpu_options);
      MPI_Status status;
      MPI_Recv(recv_tensor.data_ptr(), recv_tensor.numel(), mpi_type, i, 0, mpiComm(), &status);
      if (_gpu_aware_mpi)
        _recv_tensor[i] = recv_tensor;
      else
        _recv_tensor[i] = recv_tensor.to(device_options);
    }

  MPI_Waitall(_n_rank, send_requests.data(), MPI_STATUSES_IGNORE);

  std::vector<torch::Tensor> cat_inputs;
  cat_inputs.reserve(_n_rank);
  for (const auto & tensor : _recv_tensor)
    if (tensor.defined())
      cat_inputs.push_back(tensor);

  auto t2 = torch::cat(cat_inputs, 1);

  return torch::fft::fft(t2, c10::nullopt, 1);
}

torch::Tensor
DomainAction::ifftSlab(const torch::Tensor & t) const
{
  mooseInfoRepeated("ifftSlab");
  if (_dim == 1)
    mooseError("Unsupported mesh dimension");

  MooseTensor::printTensorInfo(t);

  // Step 1: Inverse FFT along Y direction (reciprocal space)
  // Input is in reciprocal space layout: Y-Z slabs stacked in X direction
  auto t_ifft_y = torch::fft::ifft(t, c10::nullopt, 1);
  MooseTensor::printTensorInfo(t_ifft_y);

  // Step 2: All-to-all transpose from reciprocal to real space layout
  // Need to redistribute from Y-Z slabs stacked in X to X-Z slabs stacked in Y

  const auto mpi_type = mpiTypeFromScalar(t_ifft_y.scalar_type());
  const auto cpu_options = t_ifft_y.options().device(torch::kCPU);
  const auto device_options = t_ifft_y.options();

  std::vector<MPI_Request> send_requests(_n_rank, MPI_REQUEST_NULL);
  for (const auto i : make_range(_n_rank))
  {
    auto slice = t_ifft_y.slice(1, _local_begin[1][i], _local_end[1][i]).contiguous();
    if (i == _rank)
      _recv_tensor[i] = slice;
    else
    {
      if (_gpu_aware_mpi)
        _send_tensor[i] = slice;
      else
        _send_tensor[i] = slice.to(cpu_options);

      MPI_Isend(_send_tensor[i].data_ptr(),
                _send_tensor[i].numel(),
                mpi_type,
                i,
                0,
                mpiComm(),
                &send_requests[i]);
    }
  }

  for (const auto i : make_range(_n_rank))
    if (i != _rank)
    {
      std::vector<int64_t> recv_shape;
      if (_dim == 2)
        recv_shape = {_n_local_all[0][i], _n_local_all[1][_rank]};
      else
        recv_shape = {_n_local_all[0][i], _n_local_all[1][_rank], _n_local_all[2][i]};

      auto recv_tensor = torch::empty(recv_shape, _gpu_aware_mpi ? device_options : cpu_options);
      MPI_Status status;
      MPI_Recv(recv_tensor.data_ptr(), recv_tensor.numel(), mpi_type, i, 0, mpiComm(), &status);
      if (_gpu_aware_mpi)
        _recv_tensor[i] = recv_tensor;
      else
        _recv_tensor[i] = recv_tensor.to(device_options);
    }

  MPI_Waitall(_n_rank, send_requests.data(), MPI_STATUSES_IGNORE);

  // Stack along X direction (axis 0) to get full X dimension
  std::vector<torch::Tensor> gathered;
  gathered.reserve(_n_rank);
  for (const auto & tensor : _recv_tensor)
    if (tensor.defined())
      gathered.push_back(tensor);
  auto t2 = torch::cat(gathered, 0);
  MooseTensor::printTensorInfo(t2);

  auto result = _dim == 3 ? torch::fft::ifft2(t2, c10::nullopt, {0, 2})
                          : torch::fft::ifft(t2, c10::nullopt, 0);

  auto real_result = torch::real(result);
  MooseTensor::printTensorInfo(real_result);
  return real_result;
}

torch::Tensor
DomainAction::fftPencil(const torch::Tensor & t) const
{
  if (_dim != 3)
    mooseError("Unsupported mesh dimension");
  const auto real_scalar = t.scalar_type();
  if (real_scalar != torch::kFloat32 && real_scalar != torch::kFloat64)
    mooseError("Unsupported real tensor dtype for FFT_PENCIL mode.");

  auto after_x = torch::fft::rfft(t, c10::nullopt, 0);
  auto stage1 = pencilStage1Forward(after_x);
  auto after_y = torch::fft::fft(stage1, c10::nullopt, 1);
  auto stage2 = pencilStage2Forward(after_y);
  return torch::fft::fft(stage2, c10::nullopt, 2);
}

torch::Tensor
DomainAction::ifftPencil(const torch::Tensor & t) const
{
  if (_dim != 3)
    mooseError("Unsupported mesh dimension");
  auto after_z = torch::fft::ifft(t, c10::nullopt, 2);
  auto stage2 = pencilStage2Inverse(after_z);
  auto after_y = torch::fft::ifft(stage2, c10::nullopt, 1);
  auto stage1 = pencilStage1Inverse(after_y);
  return torch::fft::irfft(stage1, _n_global[0], 0);
}

torch::Tensor
DomainAction::ifft(const torch::Tensor & t) const
{
  switch (_parallel_mode)
  {
    case ParallelMode::NONE:
      // Serial mode: use standard torch FFT functions
      switch (_dim)
      {
        case 1:
          return torch::fft::irfft(t, getShape()[0], 0);
        case 2:
          return torch::fft::irfft2(t, getShape(), {0, 1});
        case 3:
          return torch::fft::irfftn(t, getShape(), {0, 1, 2});
        default:
          mooseError("Unsupported mesh dimension");
      }

    case ParallelMode::REAL_SPACE:
      mooseError("IFFT is not available in REAL_SPACE parallel mode.");

    case ParallelMode::FFT_SLAB:
      return ifftSlab(t);

    case ParallelMode::FFT_PENCIL:
      return ifftPencil(t);
  }
  mooseError("Not implemented");
}

MPI_Comm
DomainAction::mpiComm() const
{
  return _communicator.get();
}

MPI_Datatype
DomainAction::mpiTypeFromScalar(torch::ScalarType scalar) const
{
  switch (scalar)
  {
    case torch::kFloat32:
      return MPI_FLOAT;
    case torch::kFloat64:
      return MPI_DOUBLE;
    case torch::kComplexFloat:
      return MPI_CXX_FLOAT_COMPLEX;
    case torch::kComplexDouble:
      return MPI_CXX_DOUBLE_COMPLEX;
    default:
      mooseError("Unsupported tensor dtype for MPI communication: ", static_cast<int>(scalar));
  }
  return MPI_DATATYPE_NULL;
}

torch::Tensor
DomainAction::pencilStage1Forward(const torch::Tensor & input) const
{
  const auto mpi_type = mpiTypeFromScalar(input.scalar_type());
  const auto cpu_options = input.options().device(torch::kCPU);
  const auto device_options = input.options();

  const unsigned int px = _rank % _pencil_y_partitions;
  const unsigned int group_base = _pencil_z_index[_rank] * _pencil_y_partitions;

  std::vector<MPI_Request> send_requests(_pencil_y_partitions, MPI_REQUEST_NULL);
  std::vector<torch::Tensor> send_buffers(_pencil_y_partitions);
  torch::Tensor local_chunk;

  for (unsigned int px_dest = 0; px_dest < _pencil_y_partitions; ++px_dest)
  {
    auto chunk = input
                     .slice(0,
                            _pencil_x_offsets[px_dest],
                            _pencil_x_offsets[px_dest] + _pencil_x_sizes[px_dest])
                     .contiguous();
    if (px_dest == px)
      local_chunk = chunk;
    else
    {
      if (_gpu_aware_mpi)
        send_buffers[px_dest] = chunk;
      else
        send_buffers[px_dest] = chunk.to(cpu_options);

      const auto dest_rank = group_base + px_dest;
      MPI_Isend(send_buffers[px_dest].data_ptr(),
                send_buffers[px_dest].numel(),
                mpi_type,
                dest_rank,
                10,
                mpiComm(),
                &send_requests[px_dest]);
    }
  }

  torch::Tensor result = torch::empty(
      {_pencil_x_sizes[px], static_cast<int64_t>(_n_global[1]), _n_local[2]}, device_options);

  for (unsigned int py_src = 0; py_src < _pencil_y_partitions; ++py_src)
  {
    const auto source_rank = group_base + py_src;
    torch::Tensor chunk_device;
    if (source_rank == _rank)
      chunk_device = local_chunk;
    else
    {
      std::vector<int64_t> recv_shape = {
          _pencil_x_sizes[px], _n_local_all[1][source_rank], _n_local_all[2][source_rank]};
      auto recv_tensor = torch::empty(recv_shape, _gpu_aware_mpi ? device_options : cpu_options);
      MPI_Status status;
      MPI_Recv(recv_tensor.data_ptr(),
               recv_tensor.numel(),
               mpi_type,
               source_rank,
               10,
               mpiComm(),
               &status);
      if (_gpu_aware_mpi)
        chunk_device = recv_tensor;
      else
        chunk_device = recv_tensor.to(device_options);
    }
    auto y_begin = _local_begin[1][source_rank];
    auto y_end = _local_end[1][source_rank];
    result.slice(1, y_begin, y_end).copy_(chunk_device);
  }

  MPI_Waitall(_pencil_y_partitions, send_requests.data(), MPI_STATUSES_IGNORE);
  return result;
}

torch::Tensor
DomainAction::pencilStage2Forward(const torch::Tensor & input) const
{
  const auto mpi_type = mpiTypeFromScalar(input.scalar_type());
  const auto cpu_options = input.options().device(torch::kCPU);
  const auto device_options = input.options();

  const unsigned int px = _rank % _pencil_y_partitions;
  const unsigned int y_final = _pencil_z_index[_rank];

  std::vector<MPI_Request> send_requests(_pencil_z_partitions, MPI_REQUEST_NULL);
  std::vector<torch::Tensor> send_buffers(_pencil_z_partitions);
  torch::Tensor local_chunk;

  for (unsigned int py_dest = 0; py_dest < _pencil_z_partitions; ++py_dest)
  {
    auto chunk = input
                     .slice(1,
                            _pencil_stage2_y_offsets[py_dest],
                            _pencil_stage2_y_offsets[py_dest] + _pencil_stage2_y_sizes[py_dest])
                     .contiguous();
    if (py_dest == y_final)
      local_chunk = chunk;
    else
    {
      if (_gpu_aware_mpi)
        send_buffers[py_dest] = chunk;
      else
        send_buffers[py_dest] = chunk.to(cpu_options);

      const auto dest_rank = py_dest * _pencil_y_partitions + px;
      MPI_Isend(send_buffers[py_dest].data_ptr(),
                send_buffers[py_dest].numel(),
                mpi_type,
                dest_rank,
                20,
                mpiComm(),
                &send_requests[py_dest]);
    }
  }

  torch::Tensor result = torch::empty(
      {_pencil_x_sizes[px], _pencil_stage2_y_sizes[y_final], static_cast<int64_t>(_n_global[2])},
      device_options);

  for (unsigned int z_src = 0; z_src < _pencil_z_partitions; ++z_src)
  {
    const auto source_rank = z_src * _pencil_y_partitions + px;
    torch::Tensor chunk_device;
    if (source_rank == _rank)
      chunk_device = local_chunk;
    else
    {
      std::vector<int64_t> recv_shape = {
          _pencil_x_sizes[px], _pencil_stage2_y_sizes[y_final], _n_local_all[2][source_rank]};
      auto recv_tensor = torch::empty(recv_shape, _gpu_aware_mpi ? device_options : cpu_options);
      MPI_Status status;
      MPI_Recv(recv_tensor.data_ptr(),
               recv_tensor.numel(),
               mpi_type,
               source_rank,
               20,
               mpiComm(),
               &status);
      if (_gpu_aware_mpi)
        chunk_device = recv_tensor;
      else
        chunk_device = recv_tensor.to(device_options);
    }
    auto z_begin = _local_begin[2][source_rank];
    auto z_end = _local_end[2][source_rank];
    result.slice(2, z_begin, z_end).copy_(chunk_device);
  }

  MPI_Waitall(_pencil_z_partitions, send_requests.data(), MPI_STATUSES_IGNORE);
  return result;
}

torch::Tensor
DomainAction::pencilStage2Inverse(const torch::Tensor & input) const
{
  const auto mpi_type = mpiTypeFromScalar(input.scalar_type());
  const auto cpu_options = input.options().device(torch::kCPU);
  const auto device_options = input.options();

  const unsigned int px = _rank % _pencil_y_partitions;
  const unsigned int z_idx = _pencil_z_index[_rank];

  std::vector<MPI_Request> send_requests(_pencil_z_partitions, MPI_REQUEST_NULL);
  std::vector<torch::Tensor> send_buffers(_pencil_z_partitions);
  torch::Tensor local_chunk;

  for (unsigned int z_dest = 0; z_dest < _pencil_z_partitions; ++z_dest)
  {
    const auto dest_rank = z_dest * _pencil_y_partitions + px;
    auto chunk = input.slice(2, _local_begin[2][dest_rank], _local_end[2][dest_rank]).contiguous();
    if (z_dest == z_idx)
      local_chunk = chunk;
    else
    {
      if (_gpu_aware_mpi)
        send_buffers[z_dest] = chunk;
      else
        send_buffers[z_dest] = chunk.to(cpu_options);

      MPI_Isend(send_buffers[z_dest].data_ptr(),
                send_buffers[z_dest].numel(),
                mpi_type,
                dest_rank,
                21,
                mpiComm(),
                &send_requests[z_dest]);
    }
  }

  torch::Tensor result = torch::empty(
      {_pencil_x_sizes[px], static_cast<int64_t>(_n_global[1]), _n_local[2]}, device_options);

  for (unsigned int py_src = 0; py_src < _pencil_z_partitions; ++py_src)
  {
    const auto source_rank = py_src * _pencil_y_partitions + px;
    torch::Tensor chunk_device;
    if (source_rank == _rank)
      chunk_device = local_chunk;
    else
    {
      std::vector<int64_t> recv_shape = {
          _pencil_x_sizes[px], _pencil_stage2_y_sizes[py_src], _n_local[2]};
      auto recv_tensor = torch::empty(recv_shape, _gpu_aware_mpi ? device_options : cpu_options);
      MPI_Status status;
      MPI_Recv(recv_tensor.data_ptr(),
               recv_tensor.numel(),
               mpi_type,
               source_rank,
               21,
               mpiComm(),
               &status);
      if (_gpu_aware_mpi)
        chunk_device = recv_tensor;
      else
        chunk_device = recv_tensor.to(device_options);
    }
    auto y_begin = _pencil_stage2_y_offsets[py_src];
    auto y_end = y_begin + _pencil_stage2_y_sizes[py_src];
    result.slice(1, y_begin, y_end).copy_(chunk_device);
  }

  MPI_Waitall(_pencil_z_partitions, send_requests.data(), MPI_STATUSES_IGNORE);
  return result;
}

torch::Tensor
DomainAction::pencilStage1Inverse(const torch::Tensor & input) const
{
  const auto mpi_type = mpiTypeFromScalar(input.scalar_type());
  const auto cpu_options = input.options().device(torch::kCPU);
  const auto device_options = input.options();

  const unsigned int px = _rank % _pencil_y_partitions;
  const unsigned int z_idx = _pencil_z_index[_rank];
  const unsigned int group_base = z_idx * _pencil_y_partitions;

  std::vector<MPI_Request> send_requests(_pencil_y_partitions, MPI_REQUEST_NULL);
  std::vector<torch::Tensor> send_buffers(_pencil_y_partitions);
  torch::Tensor local_chunk;

  for (unsigned int py_dest = 0; py_dest < _pencil_y_partitions; ++py_dest)
  {
    const auto dest_rank = group_base + py_dest;
    auto chunk = input.slice(1, _local_begin[1][dest_rank], _local_end[1][dest_rank]).contiguous();
    if (py_dest == px)
      local_chunk = chunk;
    else
    {
      if (_gpu_aware_mpi)
        send_buffers[py_dest] = chunk;
      else
        send_buffers[py_dest] = chunk.to(cpu_options);

      MPI_Isend(send_buffers[py_dest].data_ptr(),
                send_buffers[py_dest].numel(),
                mpi_type,
                dest_rank,
                11,
                mpiComm(),
                &send_requests[py_dest]);
    }
  }

  torch::Tensor result =
      torch::empty({static_cast<int64_t>(_n_global[0]), _n_local[1], _n_local[2]}, device_options);

  for (unsigned int px_src = 0; px_src < _pencil_y_partitions; ++px_src)
  {
    const auto source_rank = group_base + px_src;
    torch::Tensor chunk_device;
    if (source_rank == _rank)
      chunk_device = local_chunk;
    else
    {
      std::vector<int64_t> recv_shape = {_pencil_x_sizes[px_src], _n_local[1], _n_local[2]};
      auto recv_tensor = torch::empty(recv_shape, _gpu_aware_mpi ? device_options : cpu_options);
      MPI_Status status;
      MPI_Recv(recv_tensor.data_ptr(),
               recv_tensor.numel(),
               mpi_type,
               source_rank,
               11,
               mpiComm(),
               &status);
      if (_gpu_aware_mpi)
        chunk_device = recv_tensor;
      else
        chunk_device = recv_tensor.to(device_options);
    }
    auto x_begin = _pencil_x_offsets[px_src];
    auto x_end = x_begin + _pencil_x_sizes[px_src];
    result.slice(0, x_begin, x_end).copy_(chunk_device);
  }

  MPI_Waitall(_pencil_y_partitions, send_requests.data(), MPI_STATUSES_IGNORE);
  return result;
}

torch::Tensor
DomainAction::align(torch::Tensor t, unsigned int dim) const
{
  if (dim >= _dim)
    mooseError("Unsupported alignment dimension requested dimension");

  switch (_dim)
  {
    case 1:
      return t;

    case 2:
      if (dim == 0)
        return torch::unsqueeze(t, 1);
      else
        return torch::unsqueeze(t, 0);

    case 3:
      if (dim == 0)
        return t.unsqueeze(1).unsqueeze(2);
      else if (dim == 1)
        return t.unsqueeze(0).unsqueeze(2);
      else
        return t.unsqueeze(0).unsqueeze(0);

    default:
      mooseError("Unsupported mesh dimension");
  }
}

std::vector<int64_t>
DomainAction::getValueShape(std::vector<int64_t> extra_dims) const
{
  std::vector<int64_t> dims(_dim);
  for (const auto i : make_range(_dim))
    dims[i] = _n_local[i];
  dims.insert(dims.end(), extra_dims.begin(), extra_dims.end());
  return dims;
}

std::vector<int64_t>
DomainAction::getReciprocalValueShape(std::initializer_list<int64_t> extra_dims) const
{
  std::vector<int64_t> dims(_dim);
  for (const auto i : make_range(_dim))
    dims[i] = _n_reciprocal_local[i];
  dims.insert(dims.end(), extra_dims.begin(), extra_dims.end());
  return dims;
}

void
DomainAction::updateXGrid() const
{
  // TODO: add mutex to avoid thread race
  switch (_dim)
  {
    case 1:
      _x_grid = _local_axis[0];
      break;
    case 2:
      _x_grid = torch::stack({_local_axis[0].expand(_shape), _local_axis[1].expand(_shape)}, -1);
      break;
    case 3:
      _x_grid = torch::stack({_local_axis[0].expand(_shape),
                              _local_axis[1].expand(_shape),
                              _local_axis[2].expand(_shape)},
                             -1);
      break;
    default:
      mooseError("Unsupported problem dimension ", _dim);
  }
}

void
DomainAction::updateKGrid() const
{
  switch (_dim)
  {
    case 1:
      _k_grid = _local_reciprocal_axis[0];
      break;
    case 2:
      _k_grid = torch::stack({_local_reciprocal_axis[0].expand(_reciprocal_shape),
                              _local_reciprocal_axis[1].expand(_reciprocal_shape)},
                             -1);
      break;
    case 3:
      _k_grid = torch::stack({_local_reciprocal_axis[0].expand(_reciprocal_shape),
                              _local_reciprocal_axis[1].expand(_reciprocal_shape),
                              _local_reciprocal_axis[2].expand(_reciprocal_shape)},
                             -1);
      break;
    default:
      mooseError("Unsupported problem dimension ", _dim);
  }
}

void
DomainAction::updateKSquare() const
{
  _k_square = _local_reciprocal_axis[0] * _local_reciprocal_axis[0] +
              _local_reciprocal_axis[1] * _local_reciprocal_axis[1] +
              _local_reciprocal_axis[2] * _local_reciprocal_axis[2];
}

const torch::Tensor &
DomainAction::getXGrid() const
{

  // build on demand
  if (!_x_grid.defined())
    updateXGrid();

  return _x_grid;
}

const torch::Tensor &
DomainAction::getKGrid() const
{

  // build on demand
  if (!_k_grid.defined())
    updateKGrid();

  return _k_grid;
}

const torch::Tensor &
DomainAction::getKSquare() const
{
  // build on demand
  if (!_k_square.defined())
    updateKSquare();

  return _k_square;
}

void
DomainAction::getLocalBounds(unsigned int rank,
                             std::array<int64_t, 3> & begin,
                             std::array<int64_t, 3> & end) const
{
  if (rank >= _n_rank)
    mooseError("Requested local bounds for invalid rank ", rank, " (n_rank=", _n_rank, ").");

  for (const auto d : {0u, 1u, 2u})
  {
    begin[d] = _local_begin[d][rank];
    end[d] = _local_end[d][rank];
  }
}

torch::Tensor
DomainAction::sum(const torch::Tensor & t) const
{
  torch::Tensor local_sum = t.sum(_domain_dimensions, false, c10::nullopt);

  // TODO: parallel implementation
  if (comm().size() == 1)
    return local_sum;
  else
    mooseError("Sum is not implemented in parallel, yet.");
}

torch::Tensor
DomainAction::average(const torch::Tensor & t) const
{
  return sum(t) / Real(_n_global[0] * _n_global[1] * _n_global[2]);
}

int64_t
DomainAction::getNumberOfCells() const
{
  return _n_global[0] * _n_global[1] * _n_global[2];
}
