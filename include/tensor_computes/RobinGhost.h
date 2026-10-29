/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseTypes.h"

/**
 * Ghost relation of a Robin face a*u + b*du/dn = c: solve a*(u0 + ughost)/2 + b*(ughost - u0)/dx
 * = c for ughost = coefficientOnU0*u0 + constantTerm, with dx > 0 the unsigned spacing between
 * the ghost and its mirror u0. The ghost cell lies on the outward side of either face, so the
 * same expression holds on the lower and upper face. The explicit ghost fill
 * (TensorRobinBC::fillGhostCellsStatic) and the implicit solver's matrix entries (its Jacobi
 * diagonal and PetscFD::FaceGhosts on the multigrid levels) all use these, so they cannot drift
 * apart.
 */
namespace RobinGhost
{
inline Real
coefficientOnU0(Real a, Real b, Real dx)
{
  // ughost = [c - u0*(a/2 - b/dx)] / (a/2 + b/dx)  =>  coefficient on u0
  return -(a / 2.0 - b / dx) / (a / 2.0 + b / dx);
}

inline Real
constantTerm(Real a, Real b, Real c, Real dx)
{
  return c / (a / 2.0 + b / dx);
}
}
