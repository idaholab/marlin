/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannOperator.h"

class LBMNanResidual : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMNanResidual(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const int _nan_step;
  const int & _step;
};
