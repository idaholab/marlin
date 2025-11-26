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
  return batch_dims == other.batch_dims && dispatch_key == other.dispatch_key;
}

bool
TraceSchema::operator<(const TraceSchema & other) const
{
  if (batch_dims != other.batch_dims)
    return batch_dims < other.batch_dims;
  return dispatch_key < other.dispatch_key;
}

TraceSchema
TraceSchema::fromTensors(const std::vector<const torch::Tensor *> & inputs,
                         const torch::TensorOptions & options)
{
  TraceSchema schema;

  // Collect batch dimensions from all input tensors
  for (const auto * tensor : inputs)
    if (tensor && tensor->defined())
      for (int64_t i = 0; i < tensor->dim(); ++i)
        schema.batch_dims.push_back(tensor->size(i));

  // Get dispatch key from tensor options
  schema.dispatch_key = options.computeDispatchKey();

  return schema;
}
