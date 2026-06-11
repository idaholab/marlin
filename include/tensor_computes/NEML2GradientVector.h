/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "FFTGradientBase.h"

#ifdef NEML2_ENABLED
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
using NEML2GradientVectorType = neml2::Vec;
#else
using NEML2GradientVectorType = torch::Tensor;
#endif

/**
 * Gradient of a tensor field returned as a NEML2 vector.
 */
class NEML2GradientVector : public FFTGradientBase<NEML2GradientVectorType>
{
public:
  static InputParameters validParams();

  NEML2GradientVector(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor _zero;
};
