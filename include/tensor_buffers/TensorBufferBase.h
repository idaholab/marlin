/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MarlinUtils.h"
#include "MooseObject.h"
#include "InputParameters.h"
#include "DomainInterface.h"

class TensorProblem;

/**
 * Tensor wrapper arbitrary tensor value dimensions
 */
class TensorBufferBase : public MooseObject, public DomainInterface
{
public:
  static InputParameters validParams();

  TensorBufferBase(const InputParameters & parameters);

  /// assignment operator
  TensorBufferBase & operator=(const torch::Tensor & rhs);

  /// advance state, returns the new number of old states
  virtual std::size_t advanceState() = 0;

  /// clear old states
  virtual void clearStates() = 0;

  /// create a contiguous CPU copy of the current tensor (owned cells)
  virtual void makeCPUCopy() = 0;

  /// initialize the tensor
  virtual void init() {}

  /// get a raw torch tensor representation
  virtual const torch::Tensor & getRawTensor() const = 0;

  /// get a raw torch tensor representation
  virtual const torch::Tensor & getRawCPUTensor() = 0;

  /// get a view of the owned cells (excludes ghost layers)
  virtual torch::Tensor ownedView() const;

  /// expand the tensor to full dimensions
  void expand();

  const bool _reciprocal;

  const torch::IntArrayRef _domain_shape;

  const torch::TensorOptions _options;
  TensorProblem * const _tensor_problem;
};
