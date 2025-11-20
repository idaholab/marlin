/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * TensorTimeIntegrator object (this is mostly a compute object)
 */
template <typename T>
class TensorTimeIntegrator : public TensorOperator<T>
{
public:
  static InputParameters validParams();

  TensorTimeIntegrator(const InputParameters & parameters);

protected:
  const std::vector<T> & getBufferOld(const std::string & param, unsigned int max_states);
  const std::vector<T> & getBufferOldByName(const TensorInputBufferName & buffer_name,
                                            unsigned int max_states);

  const Real & _sub_dt;
};
