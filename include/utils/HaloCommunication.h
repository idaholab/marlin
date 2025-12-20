/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/**********************************************************************/

#pragma once

#include <torch/torch.h>

class DomainAction;

namespace HaloCommunication
{

/// Perform halo exchange for an arbitrary tensor using the DomainAction layout.
/// Mirrors TensorProblem::exchangeGhostLayers but works on a raw tensor
/// (temporary labels, scratch buffers, etc.).
void exchangeGhostTensor(torch::Tensor & tensor,
                         unsigned int ghost_layers,
                         const DomainAction & domain);

} // namespace HaloCommunication
