/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorNeumannBC.h"

registerMooseObject("MarlinApp", TensorNeumannBC);

InputParameters
TensorNeumannBC::validParams()
{
  InputParameters params = TensorRobinBC::validParams();
  params.set<Real>("a") = 0;
  params.set<Real>("b") = 1;
  params.suppressParameter<Real>("a");
  params.suppressParameter<Real>("b");
  params.renameParam("c", "flux", "Prescribed outward normal flux on this face (du/dn = flux).");
  params.addClassDescription(
      "Neumann boundary condition (du/dn = flux) on a non-periodic domain face.");
  return params;
}

TensorNeumannBC::TensorNeumannBC(const InputParameters & parameters) : TensorRobinBC(parameters) {}
