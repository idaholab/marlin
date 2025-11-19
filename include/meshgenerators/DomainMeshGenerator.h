/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "GeneratedMeshGenerator.h"

/**
 * Mesh generator added by the [Domain] block. Disallows renumbering.
 */
class DomainMeshGenerator : public GeneratedMeshGenerator
{
public:
  static InputParameters validParams();

  DomainMeshGenerator(const InputParameters & parameters);

  std::unique_ptr<MeshBase> generate() override;
};
