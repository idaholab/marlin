/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "Function.h"
#include "FunctionInterface.h"

/**
 * Class that represents constant function
 */
class ShiftedFunction : public Function, protected FunctionInterface
{
public:
  static InputParameters validParams();

  ShiftedFunction(const InputParameters & parameters);

  using Function::value;
  virtual Real value(Real t, const Point & p) const override;
  virtual ADReal value(const ADReal & t, const ADPoint & p) const override;

  virtual Real timeDerivative(Real t, const Point & p) const override;
  virtual RealVectorValue gradient(Real t, const Point & p) const override;

  virtual Real timeIntegral(Real t1, Real t2, const Point & p) const override;

protected:
  const Point & _delta_p;
  const Real & _delta_t;
  const Function & _function;
};
