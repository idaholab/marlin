/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TraceSchema.h"

bool
TraceSchema::operator==(const TraceSchema & other) const
{
  return tensor_ndims == other.tensor_ndims && dispatch_key == other.dispatch_key;
}

bool
TraceSchema::operator<(const TraceSchema & other) const
{
  if (tensor_ndims != other.tensor_ndims)
    return tensor_ndims < other.tensor_ndims;
  return dispatch_key < other.dispatch_key;
}

TraceSchema
TraceSchema::fromTensors(const std::vector<const torch::Tensor *> & inputs,
                         const torch::TensorOptions & options)
{
  TraceSchema schema;

  // Collect number of dimensions from all input tensors (NOT the actual sizes)
  // This allows the same trace to work with different tensor sizes
  for (const auto * tensor : inputs)
    if (tensor && tensor->defined())
      schema.tensor_ndims.push_back(tensor->dim());

  // Get dispatch key from tensor options
  schema.dispatch_key = options.computeDispatchKey();

  return schema;
}
