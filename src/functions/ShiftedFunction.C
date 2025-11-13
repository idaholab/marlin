/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ShiftedFunction.h"

registerMooseObject("MooseApp", ShiftedFunction);

InputParameters
ShiftedFunction::validParams()
{
  InputParameters params = Function::validParams();
  params.addClassDescription(
      "A function that returns a the value of another function shifted in x, y, z, and t.");
  params.addParam<Point>("shift", Point(), "The shift vector added to the sample location when evaluating the coupled function.");
  params.declareControllable("shift");
  params.addParam<Real>("delta_t", 0, "The time added to the sample time when evaluating the coupled function.");
  params.declareControllable("delta_t");
  params.addRequiredParam<FunctionName>("function", "The function to evaluate at the shifted location and time.");
  return params;
}

ShiftedFunction::ShiftedFunction(const InputParameters & parameters)
  : Function(parameters),
    FunctionInterface(this),
    _delta_p(getParam<Point>("shift")),
    _delta_t(getParam<Real>("delta_t")),
    _function(getFunction("function"))
{
}

Real
ShiftedFunction::value(Real t, const Point & p) const
{
  return _function.value(t + _delta_t, p + _delta_p);
}

ADReal
ShiftedFunction::value(const ADReal & t, const ADPoint & p) const
{
  return _function.value(t + _delta_t, p + _delta_p);
}

Real
ShiftedFunction::timeDerivative(Real t, const Point & p) const
{
  return _function.timeDerivative(t + _delta_t, p + _delta_p);
}

RealVectorValue
ShiftedFunction::gradient(Real t, const Point & p) const
{
  return _function.gradient(t + _delta_t, p + _delta_p);
}

Real
ShiftedFunction::timeIntegral(Real t1, Real t2, const Point & p) const
{
  return _function.timeIntegral(t1 + _delta_t, t2 + _delta_t, p + _delta_p);
}
