
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "DomainMeshGenerator.h"

registerMooseObject("MarlinApp", DomainMeshGenerator);

InputParameters
DomainMeshGenerator::validParams()
{
  InputParameters params = GeneratedMeshGenerator::validParams();
  return params;
}

DomainMeshGenerator::DomainMeshGenerator(const InputParameters & parameters)
  : GeneratedMeshGenerator(parameters)
{
}

std::unique_ptr<MeshBase>
DomainMeshGenerator::generate()
{
  auto mesh = GeneratedMeshGenerator::generate();
  mesh->allow_renumbering(false);
  return mesh;
}
