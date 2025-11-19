/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "UniformTensorMesh.h"

#include "MooseApp.h"

#include "libmesh/mesh_generation.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/unstructured_mesh.h"

registerMooseObject("MarlinApp", UniformTensorMesh);

InputParameters
UniformTensorMesh::validParams()
{
  InputParameters params = MooseMesh::validParams();

  MooseEnum dims("1=1 2 3");
  params.addRequiredParam<MooseEnum>("dim", dims, "The dimension of the mesh to be generated");

  params.addParam<unsigned int>("nx", 1, "Number of elements in the X direction");
  params.addParam<unsigned int>("ny", 1, "Number of elements in the Y direction");
  params.addParam<unsigned int>("nz", 1, "Number of elements in the Z direction");
  params.addParam<Real>("xmax", 1.0, "Upper X Coordinate of the generated mesh");
  params.addParam<Real>("ymax", 1.0, "Upper Y Coordinate of the generated mesh");
  params.addParam<Real>("zmax", 1.0, "Upper Z Coordinate of the generated mesh");

  params.set<bool>("allow_renumbering") = false;
  params.set<bool>("dummy_mesh") = false;
  params.suppressParameter<bool>("allow_renumbering");
  params.addClassDescription("Create a line, square, or cube mesh with uniformly spaced elements.");
  return params;
}

UniformTensorMesh::UniformTensorMesh(const InputParameters & parameters)
  : MooseMesh(parameters),
    _dim(getParam<MooseEnum>("dim")),
    _nx(getParam<unsigned int>("nx")),
    _ny(getParam<unsigned int>("ny")),
    _nz(getParam<unsigned int>("nz")),
    _xmax(getParam<Real>("xmax")),
    _ymax(getParam<Real>("ymax")),
    _zmax(getParam<Real>("zmax"))
{
  // All generated meshes are regular orthogonal meshes - until they get modified ;)
  _regular_orthogonal_mesh = true;

  // set unused dimensions to 1
  if (_dim <= 2)
    _nz = 1;
  if (_dim <= 1)
    _ny = 1;
  if (_dim == 0)
    _nx = 1;

  // Error check
  if (_nx == 0)
    paramError("nx", "Number of grid points in any direction must be greater than zero");
  if (_ny == 0)
    paramError("ny", "Number of grid points in any direction must be greater than zero");
  if (_nz == 0)
    paramError("nz", "Number of grid points in any direction must be greater than zero");
}

void
UniformTensorMesh::prepared(bool state)
{
  MooseMesh::prepared(state);

  // Fall back on scanning the mesh for coordinates instead of using input parameters for queries
  if (!state)
    mooseError("UniformTensorMesh must not be modified");
}

unsigned int
UniformTensorMesh::getElementsInDimension(unsigned int component) const
{
  switch (component)
  {
    case 0:
      return _nx;
    case 1:
      return _ny;
    case 2:
      return _nz;
    default:
      mooseError("Invalid component");
  }
}

Real
UniformTensorMesh::getMinInDimension(unsigned int) const
{
  return 0.0;
}

Real
UniformTensorMesh::getMaxInDimension(unsigned int component) const
{
  switch (component)
  {
    case 0:
      return _xmax;
    case 1:
      return _dim > 1 ? _ymax : 0.0;
    case 2:
      return _dim > 2 ? _zmax : 0.0;
    default:
      mooseError("Invalid component");
  }
}

std::unique_ptr<MooseMesh>
UniformTensorMesh::safeClone() const
{
  return _app.getFactory().copyConstruct(*this);
}

void
UniformTensorMesh::buildMesh()
{
  auto dummy = getParam<bool>("dummy_mesh");

  // Switching on MooseEnum
  switch (_dim)
  {
    // The build_XYZ mesh generation functions take an
    // UnstructuredMesh& as the first argument, hence the dynamic_cast.
    case 1:
    {

      auto elem_type = Utility::string_to_enum<ElemType>("EDGE2");
      MeshTools::Generation::build_line(dynamic_cast<UnstructuredMesh &>(getMesh()),
                                        dummy ? 1 : _nx,
                                        0.0,
                                        _xmax,
                                        elem_type,
                                        false);
      break;
    }

    case 2:
    {
      auto elem_type = Utility::string_to_enum<ElemType>("QUAD4");
      MeshTools::Generation::build_square(dynamic_cast<UnstructuredMesh &>(getMesh()),
                                          dummy ? 1 : _nx,
                                          dummy ? 1 : _ny,
                                          0.0,
                                          _xmax,
                                          0.0,
                                          _ymax,
                                          elem_type,
                                          false);
      break;
    }

    case 3:
    {
      auto elem_type = Utility::string_to_enum<ElemType>("HEX8");
      MeshTools::Generation::build_cube(dynamic_cast<UnstructuredMesh &>(getMesh()),
                                        dummy ? 1 : _nx,
                                        dummy ? 1 : _ny,
                                        dummy ? 1 : _nz,
                                        0.0,
                                        _xmax,
                                        0.0,
                                        _ymax,
                                        0.0,
                                        _zmax,
                                        elem_type,
                                        false);
      break;
    }
  }
}
