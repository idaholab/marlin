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
#include "neml2/tensors/functions/clamp.h"

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

  options.set<bool>("define_second_derivatives") = true;

  // Input variables
  options.set_input("vonmises_stress") = VariableName(STATE, "internal", "s");
  options.set("vonmises_stress").doc() = "Von Mises stress";

  options.set_input("equivalent_plastic_strain") = VariableName(STATE, "internal", "ep");
  options.set("equivalent_plastic_strain").doc() = "Equivalent plastic strain";

  options.set_input("temperature") = VariableName(FORCES, "T");
  options.set("temperature").doc() = "Temperature (optional - set use_temperature=false to disable)";

  options.set<bool>("use_temperature") = true;
  options.set("use_temperature").doc() = "Whether to include temperature effects";

  // Output
  options.set_output("flow_rate") = VariableName(STATE, "internal", "gamma_rate");
  options.set("flow_rate").doc() = "Plastic flow rate (consistency parameter rate)";

  // Johnson-Cook parameters
  options.set_parameter<TensorName<Scalar>>("A");
  options.set("A").doc() = "Reference yield stress (Pa)";

  options.set_parameter<TensorName<Scalar>>("B");
  options.set("B").doc() = "Hardening coefficient (Pa)";

  options.set_parameter<TensorName<Scalar>>("n");
  options.set("n").doc() = "Strain hardening exponent";

  options.set_parameter<TensorName<Scalar>>("C");
  options.set("C").doc() = "Rate sensitivity coefficient";

  options.set_parameter<TensorName<Scalar>>("m");
  options.set("m").doc() = "Temperature sensitivity exponent";

  options.set_parameter<TensorName<Scalar>>("reference_strain_rate");
  options.set("reference_strain_rate").doc() = "Reference strain rate (1/s)";

  options.set<double>("reference_temperature") = 300.0;
  options.set("reference_temperature").doc() = "Reference temperature (K)";

  options.set<double>("melting_temperature") = 1338.0;
  options.set("melting_temperature").doc() = "Melting temperature (K)";

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
    _T_melt(options.get<double>("melting_temperature"))
{
}

void
JohnsonCookFlowRate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  // Small value to avoid numerical issues
  const auto eps_min = Scalar::full(1e-10, _s.options());
  const auto one = Scalar::full(1.0, _s.options());
  const auto zero = Scalar::full(0.0, _s.options());

  // Strain hardening: H = A + B * ep^n
  // Use max(ep, eps_min) to avoid 0^n issues
  const auto ep_safe = Scalar(_ep) + eps_min;
  const auto ep_pow_n = pow(ep_safe, _n);
  const auto H = _A + _B * ep_pow_n;

  // Thermal softening: Theta = 1 - T*^m
  Scalar Theta;
  Scalar T_star;
  if (_T)
  {
    // T* = (T - T_ref) / (T_melt - T_ref)
    const auto dT = _T_melt - _T_ref;
    T_star = (Scalar(*_T) - _T_ref) / dT;
    // Clamp T* to [0, 0.9999] to avoid Theta = 0 or negative
    T_star = macaulay(T_star);  // max(T_star, 0)
    const auto T_star_max = Scalar::full(0.9999, _s.options());
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
  const auto sigma_y_safe = sigma_y + eps_min;  // Avoid division by zero
  const auto ratio = Scalar(_s) / sigma_y_safe;

  // Exponential argument: (ratio - 1) / C
  // Clamp to [-20, 20] for numerical stability
  const auto exp_arg = (ratio - one) / _C;
  const auto exp_arg_max = Scalar::full(20.0, _s.options());
  const auto exp_arg_min = Scalar::full(-20.0, _s.options());
  const auto exp_arg_clamped = exp_arg - macaulay(exp_arg - exp_arg_max) +
                               macaulay(exp_arg_min - exp_arg);

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
    if (_s.is_dependent())
    {
      // d(ep_dot)/d(s) = eps0 * exp(exp_arg) * H(ratio-1) * (1 / (C * sigma_y))
      // Note: We ignore the delta function from Heaviside derivative
      const auto dep_dot_ds = ep_dot / (_C * sigma_y_safe);
      _ep_dot.d(_s) = dep_dot_ds;
    }

    // Derivative with respect to equivalent plastic strain
    if (_ep.is_dependent())
    {
      // d(ep_dot)/d(ep) through the strain hardening H
      // dH/dep = B * n * ep^(n-1)
      const auto dH_dep = _B * _n * pow(ep_safe, _n - one);
      // d(sigma_y)/dep = dH/dep * Theta
      const auto dsigma_y_dep = dH_dep * Theta;
      // d(ratio)/d(ep) = -s * dsigma_y_dep / sigma_y^2
      const auto dratio_dep = -Scalar(_s) * dsigma_y_dep / (sigma_y_safe * sigma_y_safe);
      // d(exp_arg)/d(ep) = dratio_dep / C
      const auto dexp_arg_dep = dratio_dep / _C;
      // d(ep_dot)/d(ep) = ep_dot * dexp_arg_dep (ignoring Heaviside derivative)
      const auto dep_dot_dep = ep_dot * dexp_arg_dep;
      _ep_dot.d(_ep) = dep_dot_dep;
    }

    // Derivative with respect to temperature
    if (_T && _T->is_dependent())
    {
      // d(Theta)/d(T) = -m * T*^(m-1) * (1 / (T_melt - T_ref))
      const auto dT = _T_melt - _T_ref;
      const auto T_star_safe = T_star + eps_min;
      const auto dTheta_dT = -_m * pow(T_star_safe, _m - one) / dT;
      // d(sigma_y)/d(T) = H * dTheta_dT
      const auto dsigma_y_dT = H * dTheta_dT;
      // d(ratio)/d(T) = -s * dsigma_y_dT / sigma_y^2
      const auto dratio_dT = -Scalar(_s) * dsigma_y_dT / (sigma_y_safe * sigma_y_safe);
      // d(exp_arg)/d(T) = dratio_dT / C
      const auto dexp_arg_dT = dratio_dT / _C;
      // d(ep_dot)/d(T) = ep_dot * dexp_arg_dT
      const auto dep_dot_dT = ep_dot * dexp_arg_dT;
      _ep_dot.d(*_T) = dep_dot_dT;
    }
  }

  // Second derivatives (for consistent tangent if needed)
  if (d2out_din2)
  {
    // Second derivatives can be implemented here if needed for higher-order convergence
    // For now, we rely on first derivatives being sufficient for Newton convergence
  }
}
} // namespace neml2

#endif // NEML2_ENABLED
