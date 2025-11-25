/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#ifdef NEML2_ENABLED

#include "neml2/models/Model.h"

namespace neml2
{
class Scalar;

/**
 * @brief Johnson-Cook plastic flow rate model
 *
 * Computes the equivalent plastic strain rate using the inverted Johnson-Cook equation:
 * \f[
 *   \dot{\varepsilon}_p = \dot{\varepsilon}_0 \exp\left(\frac{\sigma_{vm} / (\sigma_y \Theta) - 1}{C}\right) H(\sigma_{vm} - \sigma_y \Theta)
 * \f]
 *
 * where:
 * - \f$ \sigma_y = A + B \varepsilon_p^n \f$ is the strain hardening
 * - \f$ \Theta = 1 - T^{*m} \f$ is the thermal softening
 * - \f$ T^* = (T - T_{ref}) / (T_{melt} - T_{ref}) \f$ is the homologous temperature
 *
 * This formulation computes strain rate from stress, avoiding the circular dependency
 * that would arise from directly using the Johnson-Cook yield stress formulation.
 */
class JohnsonCookFlowRate : public Model
{
public:
  static OptionSet expected_options();

  JohnsonCookFlowRate(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Von Mises stress (input)
  const Variable<Scalar> & _s;

  /// Equivalent plastic strain (input)
  const Variable<Scalar> & _ep;

  /// Temperature (input, optional)
  const Variable<Scalar> * _T;

  /// Plastic strain rate (output)
  Variable<Scalar> & _ep_dot;

  /// Reference yield stress (A parameter)
  const Scalar & _A;

  /// Hardening coefficient (B parameter)
  const Scalar & _B;

  /// Strain hardening exponent (n parameter)
  const Scalar & _n;

  /// Rate sensitivity coefficient (C parameter)
  const Scalar & _C;

  /// Temperature sensitivity exponent (m parameter)
  const Scalar & _m;

  /// Reference strain rate
  const Scalar & _eps0;

  /// Reference temperature
  const double _T_ref;

  /// Melting temperature
  const double _T_melt;
};
} // namespace neml2

#endif // NEML2_ENABLED
