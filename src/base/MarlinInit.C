/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MarlinInit.h"
#include "MarlinApp.h"

MarlinInit::MarlinInit(int argc, char * argv[], MPI_Comm COMM_WORLD_IN)
  : MooseInit(argc, argv, COMM_WORLD_IN)
{
  for (const auto i : make_range(argc - 1))
    if (std::string(argv[i]) == "--libtorch-device")
      MarlinApp::setTorchDeviceStatic(argv[i + 1], {});
}
