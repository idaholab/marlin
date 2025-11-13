/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

// MOOSE includes
#include "MooseEnum.h"
#include "MoosePartitioner.h"

class MooseMesh;

namespace libMesh
{
class SubdomainPartitioner;
}

/**
 * Partitions a mesh using a regular grid.
 */
class DomainPartitioner : public MoosePartitioner
{
public:
  DomainPartitioner(const InputParameters & params);
  virtual ~DomainPartitioner();

  static InputParameters validParams();

  virtual std::unique_ptr<Partitioner> clone() const override;

protected:
  virtual void _do_partition(MeshBase & mesh, const unsigned int n) override;

  MooseMesh & _mesh;
};
