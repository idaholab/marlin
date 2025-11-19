/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseInit.h"

class MarlinInit : public MooseInit
{
public:
  MarlinInit(int argc, char * argv[], MPI_Comm COMM_WORLD_IN = MPI_COMM_WORLD);
};
