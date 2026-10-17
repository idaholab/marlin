/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorDirichletBC.h"

registerMooseObject("MarlinApp", TensorDirichletBC);

InputParameters
TensorDirichletBC::validParams()
{
  InputParameters params = TensorRobinBC::validParams();
  params.set<Real>("a") = 1;
  params.set<Real>("b") = 0;
  params.suppressParameter<Real>("a");
  params.suppressParameter<Real>("b");
  params.renameParam("c", "value", "Prescribed value of the field on this face (u = value).");
  params.addClassDescription("Dirichlet boundary condition (u = value) on a non-periodic domain "
                             "face.");
  return params;
}

TensorDirichletBC::TensorDirichletBC(const InputParameters & parameters) : TensorRobinBC(parameters)
{
}
