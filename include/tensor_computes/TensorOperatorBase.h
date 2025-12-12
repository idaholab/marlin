/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseObject.h"
#include "MarlinTypes.h"
#include "TensorProblem.h"
#include "DependencyResolverInterface.h"
#include "MarlinConstantInterface.h"
#include "TraceableUtils.h"

#include <map>
#include <torch/torch.h>

class DomainAction;
class TensorBufferBase;

/**
 * TensorOperatorBase object
 */
class TensorOperatorBase : public MooseObject,
                           public DependencyResolverInterface,
                           public MarlinConstantInterface
{
public:
  static InputParameters validParams();

  TensorOperatorBase(const InputParameters & parameters);

  virtual const std::set<std::string> & getRequestedItems() override { return _requested_buffers; }
  virtual const std::set<std::string> & getSuppliedItems() override { return _supplied_buffers; }

  /// Helper to recursively update dependencies for grouped operators
  virtual void updateDependencies() {}

  /// perform the computation
  virtual void computeBuffer() = 0;

  /// perform the computation in real space
  virtual void realSpaceComputeBuffer();

  /// called  after all objects have been constructed (before dependency resolution)
  virtual void init() {}

  /// called  after all objects have been constructed (after dependency resolution)
  virtual void check() {}

  /// called if the simulation cell dimensions change
  virtual void gridChanged() {}

  /// Whether this compute supports JIT tracing. Override and return false for computes that:
  /// - Use random number generation with changing seeds
  /// - Have data-dependent control flow (if statements based on tensor values)
  /// - Call external libraries that aren't traceable
  /// - Modify global state
  /// - Use FFT operations in parallel mode (MPI communication is not traceable)
  virtual bool supportsJIT() const { return true; }

public:
  /// Helper for computes that use FFT: returns true if FFT requires MPI (not JIT-traceable)
  bool usesParallelFFT() const;

  /**
   * Get a tensor dimension as a traceable size.
   * During JIT tracing, returns a symbolic reference that evaluates at runtime.
   * This enables traces to work with different tensor sizes.
   */
  TraceableSize getTraceableSize(const torch::Tensor & tensor, int64_t dim) const
  {
    return TraceableUtils::getTraceableSize(tensor, dim);
  }

  /**
   * Extract tensor shape as traceable sizes.
   * @param tensor The tensor to get shape from
   * @param ndim Number of dimensions to extract (0 = all)
   */
  TraceableTensorShape getTraceableShape(const torch::Tensor & tensor, int64_t ndim = 0) const
  {
    return TraceableUtils::extractTraceableSizes(tensor, ndim);
  }
  template <typename T = torch::Tensor>
  const T & getInputBuffer(const std::string & param, unsigned int ghost_layers = 0);

  template <typename T = torch::Tensor>
  const T & getInputBufferByName(const TensorInputBufferName & buffer_name,
                                 unsigned int ghost_layers = 0);

  template <typename T = torch::Tensor>
  T & getOutputBuffer(const std::string & param);

  template <typename T = torch::Tensor>
  T & getOutputBufferByName(const TensorOutputBufferName & buffer_name);

  TensorOperatorBase & getCompute(const std::string & param_name);
  TensorBufferBase & getBufferBase(const TensorInputBufferName & buffer_name);

  const std::map<std::string, unsigned int> & getInputGhostLayers() const
  {
    return _input_buffer_ghost_layers;
  }

  std::set<std::string> _requested_buffers;
  std::set<std::string> _supplied_buffers;
  std::map<std::string, unsigned int> _input_buffer_ghost_layers;

  TensorProblem & _tensor_problem;
  const DomainAction & _domain;

  /// axes
  const torch::Tensor &_x, &_y, &_z;

  /// reciprocal axes
  const torch::Tensor &_i, &_j, &_k;

  /// Imaginary unit i
  const torch::Tensor _imaginary;

  /// substep time
  const Real & _time;

  /// problem dimension
  const unsigned int & _dim;
};

template <typename T>
const T &
TensorOperatorBase::getInputBuffer(const std::string & param, unsigned int ghost_layers)
{
  return getInputBufferByName<T>(getParam<TensorInputBufferName>(param), ghost_layers);
}

template <typename T>
const T &
TensorOperatorBase::getInputBufferByName(const TensorInputBufferName & buffer_name,
                                         unsigned int ghost_layers)
{
  _requested_buffers.insert(buffer_name);
  _input_buffer_ghost_layers[buffer_name] =
      std::max(_input_buffer_ghost_layers[buffer_name], ghost_layers);
  _tensor_problem.registerGhostLayerRequest(buffer_name, ghost_layers);
  return _tensor_problem.getBuffer<T>(buffer_name, ghost_layers);
}

template <typename T>
T &
TensorOperatorBase::getOutputBuffer(const std::string & param)
{
  return getOutputBufferByName<T>(getParam<TensorOutputBufferName>(param));
}

template <typename T>
T &
TensorOperatorBase::getOutputBufferByName(const TensorOutputBufferName & buffer_name)
{
  _supplied_buffers.insert(buffer_name);
  return _tensor_problem.getBuffer<T>(buffer_name);
}

inline TensorBufferBase &
TensorOperatorBase::getBufferBase(const TensorInputBufferName & buffer_name)
{
  _requested_buffers.insert(buffer_name);
  return _tensor_problem.getBufferBase(buffer_name);
}
