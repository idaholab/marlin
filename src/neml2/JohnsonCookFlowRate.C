/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#ifdef NEML2_ENABLED

#include "JohnsonCookFlowRate.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/exp.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/heaviside.h"
#include "neml2/tensors/functions/macaulay.h"

namespace neml2
{
register_NEML2_object(JohnsonCookFlowRate);

OptionSet
JohnsonCookFlowRate::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Johnson-Cook plastic flow rate model. Computes equivalent plastic strain rate from "
      "stress using the inverted Johnson-Cook equation: "
      "\\f$ \\dot{\\varepsilon}_p = \\dot{\\varepsilon}_0 \\exp\\left(\\frac{\\sigma_{vm} / "
      "(\\sigma_y \\Theta) - 1}{C}\\right) \\f$ "
      "where \\f$ \\sigma_y = A + B \\varepsilon_p^n \\f$ and \\f$ \\Theta = 1 - T^{*m} \\f$.";

  // Input variables
  options.add_input("vonmises_stress", "Von Mises stress");
  options.add_input("equivalent_plastic_strain", "Equivalent plastic strain");
  options.add_optional_input("temperature",
                             "Temperature (optional - set use_temperature=false to disable)");

  options.add<bool>("use_temperature", true, "Whether to include temperature effects");

  // Output
  options.add_output("flow_rate", "Plastic flow rate (consistency parameter rate)");

  // Johnson-Cook parameters
  options.add_parameter<Scalar>("A", "Reference yield stress (Pa)");
  options.add_parameter<Scalar>("B", "Hardening coefficient (Pa)");
  options.add_parameter<Scalar>("n", "Strain hardening exponent");
  options.add_parameter<Scalar>("C", "Rate sensitivity coefficient");
  options.add_parameter<Scalar>("m", "Temperature sensitivity exponent");
  options.add_parameter<Scalar>("reference_strain_rate", "Reference strain rate (1/s)");

  options.add<double>("reference_temperature", 300.0, "Reference temperature (K)");
  options.add<double>("melting_temperature", 1338.0, "Melting temperature (K)");
  options.add<double>(
      "initial_plastic_strain",
      0.0,
      "Prior accumulated equivalent plastic strain from the specimen temper (e.g. ~0.37 for H04 "
      "full-hard copper). Added to the evolving plastic strain in the hardening term, "
      "sigma_y = A + B (ep + ep0)^n, so the material starts pre-hardened. Default 0 = annealed.");
  options.add<double>(
      "max_homologous_temperature",
      0.9999,
      "Upper clamp on the homologous temperature T* in the thermal softening term "
      "Theta = 1 - T*^m. The default keeps a vanishing strength floor; lowering it (e.g. 0.95) "
      "retains a finite flow-stress floor near melt, which regularizes the adiabatic "
      "heat-soften-strain feedback that otherwise localizes in a single element when conduction "
      "is not modeled.");

  return options;
}

JohnsonCookFlowRate::JohnsonCookFlowRate(const OptionSet & options)
  : Model(options),
    _s(declare_input_variable<Scalar>("vonmises_stress")),
    _ep(declare_input_variable<Scalar>("equivalent_plastic_strain")),
    _T(options.get<bool>("use_temperature") ? &declare_input_variable<Scalar>("temperature")
                                            : nullptr),
    _ep_dot(declare_output_variable<Scalar>("flow_rate")),
    _A(declare_parameter<Scalar>("A", "A", /*allow_nonlinear=*/true)),
    _B(declare_parameter<Scalar>("B", "B", /*allow_nonlinear=*/true)),
    _n(declare_parameter<Scalar>("n", "n", /*allow_nonlinear=*/true)),
    _C(declare_parameter<Scalar>("C", "C", /*allow_nonlinear=*/true)),
    _m(declare_parameter<Scalar>("m", "m", /*allow_nonlinear=*/true)),
    _eps0(declare_parameter<Scalar>("eps0", "reference_strain_rate", /*allow_nonlinear=*/true)),
    _T_ref(options.get<double>("reference_temperature")),
    _T_melt(options.get<double>("melting_temperature")),
    _T_star_max(options.get<double>("max_homologous_temperature")),
    _ep0(options.get<double>("initial_plastic_strain"))
{
}

void
JohnsonCookFlowRate::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // Small value to avoid numerical issues
  const auto eps_min = Scalar::full(1e-10, _s.options());
  const auto one = Scalar::full(1.0, _s.options());
  const auto zero = Scalar::full(0.0, _s.options());

  // Strain hardening: H = A + B * (ep + ep0)^n, where ep0 is the prior cold-work
  // strain from the temper (0 = annealed). max(ep,0) + ep0 + eps_min avoids 0^n.
  const auto ep0 = Scalar::full(_ep0, _s.options());
  const auto ep_positive = macaulay(_ep());
  const auto ep_safe = ep_positive + ep0 + eps_min;
  const auto ep_pow_n = pow(ep_safe, _n);
  const auto H = _A + _B * ep_pow_n;

  // Thermal softening: Theta = 1 - T*^m
  Scalar Theta;
  Scalar T_star;
  if (_T)
  {
    // T* = (T - T_ref) / (T_melt - T_ref)
    const auto dT = _T_melt - _T_ref;
    T_star = ((*_T)() - _T_ref) / dT;
    // Clamp T* to [0, max_homologous_temperature] to avoid Theta <= 0 (and, with a lowered
    // clamp, to keep a finite flow-stress floor near melt)
    T_star = macaulay(T_star); // max(T_star, 0)
    const auto T_star_max = Scalar::full(_T_star_max, _s.options());
    // Use torch minimum
    const auto T_star_clamped = T_star - macaulay(T_star - T_star_max);
    Theta = one - pow(T_star_clamped, _m);
  }
  else
  {
    // Isothermal: Theta = 1
    Theta = one;
    T_star = zero;
  }

  // Yield strength: sigma_y = H * Theta
  const auto sigma_y = H * Theta;

  // Stress ratio: ratio = sigma_vm / sigma_y
  const auto sigma_y_safe = sigma_y + eps_min; // Avoid division by zero
  const auto ratio = _s() / sigma_y_safe;

  // Exponential argument: (ratio - 1) / C
  // Clamp to [-700, 700] only to prevent IEEE double overflow (exp(709) ~ DBL_MAX).
  // The gradient is NOT zeroed at the clamp boundary — we use the straight-through
  // estimator so Newton always sees the correct sensitivity direction.
  const auto exp_arg = (ratio - one) / _C;
  const auto exp_arg_max = Scalar::full(700.0, _s.options());
  const auto exp_arg_min = Scalar::full(-700.0, _s.options());
  const auto exp_arg_clamped =
      exp_arg - macaulay(exp_arg - exp_arg_max) + macaulay(exp_arg_min - exp_arg);

  // Flow rate: ep_dot = eps0 * exp(exp_arg) * H(ratio - 1)
  // Using Heaviside to ensure no plastic flow when below yield
  const auto exp_val = exp(exp_arg_clamped);
  const auto H_yield = heaviside(ratio - one);
  const auto ep_dot = _eps0 * exp_val * H_yield;

  if (out)
    _ep_dot = ep_dot;

  // Compute derivatives for Newton solver
  if (dout_din)
  {
    // Derivative with respect to von Mises stress
    {
      // d(ep_dot)/d(s) = eps0 * exp(exp_arg) * H(ratio-1) * (1 / (C * sigma_y))
      // Note: We ignore the delta function from Heaviside derivative
      const auto dep_dot_ds = ep_dot / (_C * sigma_y_safe);
      _ep_dot.d(_s) = dep_dot_ds;
    }

    // Derivative with respect to equivalent plastic strain
    {
      // d(ep_dot)/d(ep) through the strain hardening H
      // dH/dep = B * n * ep^(n-1)
      const auto dH_dep = _B * _n * heaviside(_ep()) * pow(ep_safe, _n - one);
      // d(sigma_y)/dep = dH/dep * Theta
      const auto dsigma_y_dep = dH_dep * Theta;
      // d(ratio)/d(ep) = -s * dsigma_y_dep / sigma_y^2
      const auto dratio_dep = -_s() * dsigma_y_dep / (sigma_y_safe * sigma_y_safe);
      // d(exp_arg)/d(ep) = dratio_dep / C
      const auto dexp_arg_dep = dratio_dep / _C;
      // d(ep_dot)/d(ep) = ep_dot * dexp_arg_dep (ignoring Heaviside derivative)
      const auto dep_dot_dep = ep_dot * dexp_arg_dep;
      _ep_dot.d(_ep) = dep_dot_dep;
    }

    // Derivative with respect to temperature
    if (_T)
    {
      // d(Theta)/d(T) = -m * T*^(m-1) * (1 / (T_melt - T_ref))
      const auto dT = _T_melt - _T_ref;
      const auto T_star_safe = T_star + eps_min;
      const auto dTheta_dT = -_m * pow(T_star_safe, _m - one) / dT;
      // d(sigma_y)/d(T) = H * dTheta_dT
      const auto dsigma_y_dT = H * dTheta_dT;
      // d(ratio)/d(T) = -s * dsigma_y_dT / sigma_y^2
      const auto dratio_dT = -_s() * dsigma_y_dT / (sigma_y_safe * sigma_y_safe);
      // d(exp_arg)/d(T) = dratio_dT / C
      const auto dexp_arg_dT = dratio_dT / _C;
      // d(ep_dot)/d(T) = ep_dot * dexp_arg_dT
      const auto dep_dot_dT = ep_dot * dexp_arg_dT;
      _ep_dot.d(*_T) = dep_dot_dT;
    }
  }
}
} // namespace neml2

#endif // NEML2_ENABLED
