/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOutput.h"
#include "pugixml.h"
#include <thread>

#ifdef LIBMESH_HAVE_HDF5
#include "hdf5.h"
#endif

/**
 * XDMF (XML + binary/hdf5) file output
 */
class XDMFTensorOutput : public TensorOutput
{
public:
  static InputParameters validParams();

  XDMFTensorOutput(const InputParameters & parameters);
  ~XDMFTensorOutput();

  virtual void init() override;
  virtual void prepareForOutput() override;

protected:
  virtual void output() override;

  torch::Tensor extendTensor(torch::Tensor tensor);
  torch::Tensor upsampleTensor(torch::Tensor tensor);

private:
  enum class OutputMode
  {
    CELL,
    NODE,
    OVERSIZED_NODAL
  };

  void writeLocalData();
  void writeSerialXMF();
  void writeParallelXMF();
  std::vector<std::string> buildAttributeNames(const TensorInputBufferName & buffer_name,
                                               int64_t num_fields) const;
  unsigned int mappedAxis(unsigned int axis) const;
  std::vector<int64_t> localCellCounts(unsigned int rank) const;
  std::vector<int64_t> localNodeCounts(unsigned int rank) const;
  std::vector<Real> localOrigin(unsigned int rank) const;
  std::vector<Real> localSpacing() const;
  std::string dimsToString(const std::vector<int64_t> & dims) const;
  std::string rankTag(unsigned int rank) const;
  std::string hdf5FileName(unsigned int rank) const;
  std::string binaryFileName(const std::string & setname, unsigned int rank) const;

  /// mesh dimension
  const unsigned int _dim;

  /// frame counter
  std::size_t _frame;

  /// communicator info
  const unsigned int _rank;
  const unsigned int _n_rank;
  const bool _is_parallel;

  /// transpose tensors before outputting to counter a Paraview XDMF reader ideosyncracy
  const bool _transpose;

  /// whether the tensor uses Cell or Node output
  std::map<std::string, OutputMode> _output_mode;

  /// precomputed global data dimensions (serial)
  std::array<std::vector<std::size_t>, 2> _ndata;
  std::array<std::string, 2> _data_grid;
  std::vector<std::size_t> _nnode;
  std::string _node_grid;
  std::string _geometry_type;
  std::vector<std::pair<std::array<int64_t, 3>, std::array<int64_t, 3>>> _cached_local_bounds;

  /// TODO: precomputed origin (parallel) to avoid garbage values in origin on last output
  std::vector<Real> _origin;

  /// xml document references (root rank only in parallel)
  pugi::xml_document _doc;
  pugi::xml_node _tgrid;

#ifdef LIBMESH_HAVE_HDF5
  const bool _enable_hdf5;

  /// HDF5 file name (rank-specific)
  const std::string _hdf5_name;

  /// HDF5 file handle
  hid_t _hdf5_file_id;
#endif
};

/*

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>

void outputScalarFields(const torch::Tensor& tensor, int spatial_dim, const std::string&
filename_prefix) { auto sizes = tensor.sizes(); int total_dims = sizes.size();

    if (spatial_dim < 1 || spatial_dim > 3) {
        std::cerr << "spatial_dim must be between 1 and 3." << std::endl;
        return;
    }

    int scalar_dims = total_dims - spatial_dim;
    if (scalar_dims < 0) {
        std::cerr << "Tensor has fewer dimensions than specified spatial_dim." << std::endl;
        return;
    }

    int num_scalar_fields = 1;
    for (int i = spatial_dim; i < total_dims; ++i) {
        num_scalar_fields *= sizes[i];
    }

    std::vector<int64_t> reshape_sizes(sizes.begin(), sizes.begin() + spatial_dim);
    reshape_sizes.push_back(num_scalar_fields);
    torch::Tensor reshaped = tensor.reshape(reshape_sizes);

    for (int field_idx = 0; field_idx < num_scalar_fields; ++i) {
        torch::Tensor scalar_field = reshaped.select(spatial_dim, field_idx);

        std::string filename = filename_prefix + "_field_" + std::to_string(i) + ".txt";
        std::ofstream outfile(filename);

        if (!outfile.is_open()) {
            std::cerr << "Failed to open file " << filename << std::endl;
            continue;
        }

        auto accessor = scalar_field.accessor<float, spatial_dim>();

        if (spatial_dim == 1) {
            for (int64_t x = 0; x < scalar_field.size(0); ++x) {
                outfile << scalar_field[x].item<float>() << "\n";
            }
        } else if (spatial_dim == 2) {
            for (int64_t i = 0; i < scalar_field.size(0); ++i) {
                for (int64_t j = 0; j < scalar_field.size(1); ++j) {
                    outfile << scalar_field[i][j].item<float>() << " ";
                }
                outfile << "\n";
            }
        } else if (spatial_dim == 3) {
            for (int64_t i = 0; i < scalar_field.size(0); ++i) {
                for (int64_t j = 0; j < scalar_field.size(1); ++j) {
                    for (int64_t k = 0; k < scalar_field.size(2); ++k) {
                        outfile << scalar_field[i][j][k].item<float>() << " ";
                    }
                    outfile << "\n";
                }
                outfile << "\n";
            }
        }

        outfile.close();
        std::cout << "Saved Scalar Field " << field_idx << " to " << filename << std::endl;
    }
}

int main() {
    torch::Tensor tensor = torch::rand({20, 20, 20, 3});  // example 3D vector field
    outputScalarFields(tensor, 3, "tensor_output");
    return 0;
}

*/
