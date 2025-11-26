/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MarlinConstantInterface.h"
#include "InputParameters.h"

MarlinConstantInterface::MarlinConstantInterface(const InputParameters & params)
  : _params(params),
    _sci_tensor_problem(*params.getCheckedPointerParam<TensorProblem *>("_tensor_problem"))
{
}
