/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "SplitOperatorBase.h"

/**
 * Adams-Bashforth-Moulton semi-implicit/explicit solver for coupled systems
 * with a full linear operator L. Uses a batched torch linear solve per
 * reciprocal-space wavenumber to apply (I - dt L)^{-1}.
 */
class AdamsBashforthMoultonCoupled : public SplitOperatorBase
{
public:
  static InputParameters validParams();

  AdamsBashforthMoultonCoupled(const InputParameters & parameters);

protected:
  // Max order supported (up to ABM5)
  static constexpr std::size_t max_order = 5;

  virtual void substep() override;

  // configuration
  unsigned int _substeps;
  std::size_t _predictor_order;
  std::size_t _corrector_order;
  std::size_t _corrector_steps;
  bool _assume_symmetric;

  // off-diagonal specification (i -> row, j -> col)
  std::vector<std::pair<unsigned int, unsigned int>> _L_offdiag_indices;
  std::vector<TensorInputBufferName> _L_offdiag_names;
  std::vector<const torch::Tensor *> _L_offdiag_buffer;

  // references to substep state
  Real & _sub_dt;
  Real & _sub_time;
};
