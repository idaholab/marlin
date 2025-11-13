/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LatticeBoltzmannStencilBase.h"

InputParameters
LatticeBoltzmannStencilBase::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.addClassDescription("LB Stencil object.");
  params.registerBase("LBStencil");
  return params;
}

LatticeBoltzmannStencilBase::LatticeBoltzmannStencilBase(const InputParameters & parameters)
  : MooseObject(parameters)
{
}
