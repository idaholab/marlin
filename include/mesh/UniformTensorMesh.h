/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseMesh.h"
#include "DomainInterface.h"

/**
 * Mesh generated from parameters
 */
class UniformTensorMesh : public MooseMesh
{
public:
  static InputParameters validParams();

  UniformTensorMesh(const InputParameters & parameters);
  UniformTensorMesh(const UniformTensorMesh & /* other_mesh */) = default;

  // No copy
  UniformTensorMesh & operator=(const UniformTensorMesh & other_mesh) = delete;

  virtual std::unique_ptr<MooseMesh> safeClone() const override;

  unsigned int getDim() const { return _dim; }

  virtual void buildMesh() override;
  unsigned int getElementsInDimension(unsigned int component) const;
  virtual Real getMinInDimension(unsigned int component) const override;
  virtual Real getMaxInDimension(unsigned int component) const override;
  virtual void prepared(bool state) override;

protected:
  /// The dimension of the mesh
  MooseEnum _dim;

  /// Number of elements in x, y, z direction
  unsigned int _nx, _ny, _nz;

  /// The max values for x,y,z component
  Real _xmax, _ymax, _zmax;
};
