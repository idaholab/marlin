/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

class LatticeBoltzmannProblem;
class LatticeBoltzmannStencilBase;

/**
 * LatticeBoltzmannOperator object
 */
class LatticeBoltzmannOperator : public TensorOperator<>
{
public:
  static InputParameters validParams();

  LatticeBoltzmannOperator(const InputParameters & parameters);
  torch::Tensor ownedView(const torch::Tensor &);
  virtual void realSpaceComputeBuffer() override;

protected:
  LatticeBoltzmannProblem & _lb_problem;
  const LatticeBoltzmannStencilBase & _stencil;

  // owned copy of output tensor
  torch::Tensor _u_owned;

  const torch::Tensor _ex;
  const torch::Tensor _ey;
  const torch::Tensor _ez;
  const torch::Tensor _w;

  const std::vector<int64_t> & _shape;
  const std::vector<int64_t> & _shape_q;

  // radius of the stencil
  const unsigned int _radius;
};
