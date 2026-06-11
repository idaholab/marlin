/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperatorBase.h"
#include "GrainRemap.h"

template <typename T>
class TensorBuffer;

/**
 * Tracks grains (connected components of multi-order-parameter phase fields),
 * maintains persistent grain identities across timesteps, and remaps grains to
 * different order parameters when grains sharing an order parameter approach
 * each other closer than the exclusion distance (halo_width).
 *
 * In REAL_SPACE parallel mode the tracker stitches grains across rank boundaries
 * (and periodic boundaries) using ghost layer exchange; every rank computes the
 * identical global grain list. In other (spectral) modes the tracker supports
 * serial runs with periodicity handled internally.
 */
class GrainTracker : public TensorOperatorBase
{
public:
  static InputParameters validParams();

  GrainTracker(const InputParameters & parameters);

  virtual void computeBuffer() override;
  virtual void check() override;
  virtual bool supportsJIT() const override { return false; }

  /// number of grains found in the last tracking step
  std::size_t getGrainCount() const { return _grains.size(); }
  /// cumulative number of grain->order parameter moves performed
  std::size_t getRemapCount() const { return _total_remaps; }
  /// number of unresolvable color conflicts in the last tracking step
  std::size_t getConflictCount() const { return _last_conflicts; }
  /// grain metadata of the last tracking step
  const std::vector<GrainRemap::GrainMeta> & getGrains() const { return _grains; }

protected:
  void trackAndRemap();

  /// gather cropped per-order-parameter views (owned + active halo ring) into the buffers
  std::vector<torch::Tensor> cropOpBuffers(int64_t buffer_pad, int64_t hw);

  /// names of the per-order-parameter buffers
  const std::vector<TensorInputBufferName> _op_names;
  /// phase field threshold for grain detection
  const Real _threshold;
  /// exclusion / remap halo distance in cells
  const unsigned int _halo_width;
  /// face-only or full (corner) connectivity
  const MooseEnum _connectivity;
  /// centroid matching tolerance in cells
  const Real _tracking_tolerance;
  /// volume ratio guard for persistence matching
  const Real _tracking_volume_ratio;
  /// run the tracker every `interval` executions
  const unsigned int _interval;
  /// remap old states (time integrator history) along with the current state
  const bool _remap_old_states;
  /// error or warn when grains cannot be assigned conflict-free order parameters
  const MooseEnum _on_conflict;
  /// optional output buffer holding the persistent grain id per cell
  torch::Tensor * _grain_id_buffer;

  /// non-const references to the order parameter buffer tensors (remapped in place)
  std::vector<torch::Tensor *> _op_tensors;
  /// buffer objects, used for old-state (history) access
  std::vector<TensorBuffer<torch::Tensor> *> _op_buffers;

  /// grain metadata of the last tracking step (identical on all ranks)
  std::vector<GrainRemap::GrainMeta> _grains;
  /// number of executions
  unsigned int _execution_count;
  /// cumulative remap moves
  std::size_t _total_remaps;
  /// conflicts in last step
  std::size_t _last_conflicts;
};
