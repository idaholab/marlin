/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMD2Q9.h"

registerMooseObject("MarlinApp", LBMD2Q9);

InputParameters
LBMD2Q9::validParams()
{
  InputParameters params = LatticeBoltzmannStencilBase::validParams();
  params.addClassDescription("LBMD2Q9 Stencil object.");
  return params;
}

LBMD2Q9::LBMD2Q9(const InputParameters & parameters) : LatticeBoltzmannStencilBase(parameters)
{
  _q = 9;
  _ex = torch::tensor({0, 1, 0, -1, 0, 1, -1, -1, 1}, MooseTensor::intTensorOptions());
  _ey = torch::tensor({0, 0, 1, 0, -1, 1, 1, -1, -1}, MooseTensor::intTensorOptions());
  _ez = torch::tensor({0, 0, 0, 0, 0, 0, 0, 0, 0}, MooseTensor::intTensorOptions());

  _weights = torch::tensor({4.0 / 9.0,
                            1.0 / 9.0,
                            1.0 / 9.0,
                            1.0 / 9.0,
                            1.0 / 9.0,
                            1.0 / 36.0,
                            1.0 / 36.0,
                            1.0 / 36.0,
                            1.0 / 36.0},
                           MooseTensor::floatTensorOptions());

  _op = torch::tensor({0, 3, 4, 1, 2, 7, 8, 5, 6}, MooseTensor::intTensorOptions());

  // transformation matrix
  _M = torch::tensor({{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                      {-4.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0},
                      {4.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0},
                      {0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0},
                      {0.0, -2.0, 0.0, 2.0, 0.0, 1.0, -1.0, -1.0, 1.0},
                      {0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0},
                      {0.0, 0.0, -2.0, 0.0, 2.0, 1.0, 1.0, -1.0, -1.0},
                      {0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0},
                      {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0}},
                     MooseTensor::floatTensorOptions());
  _M_inv = at::linalg_inv(_M);

  // relaxation matrix
  _S = torch::diag(torch::tensor({1.0 / 1.0,     // tau_rho : mass
                                  1.0 / 1.1,     // tau_e : energy
                                  1.0 / 1.2,     // tau_epsilon : energy square
                                  1.0 / 1.0000,  // tau_j : momentum
                                  1.0 / 1.0000,  // tau_q : slip velocity
                                  1.0 / 1.0000,  // tau_j : momentum
                                  1.0 / 1.0000,  // tau_q : slip velocity
                                  1.0 / 1.0000,  // tau_s : shear viscosity
                                  1.0 / 1.0000}, // tau_s : shear viscosity
                                 MooseTensor::floatTensorOptions()));
  // indices where relaxation parameter related to kinematic viscosity (i.e. shear stress) is
  // located
  _id_kinematic_visc = torch::tensor({7, 8}, MooseTensor::intTensorOptions());

  /**
   * incoming unknown distribution functions at every face
   * the opposite faces can be determined using _op vector
   * E.g. the opposite of _top[0] is _bottom[0] = _op[top[0]]
   */
  _left = torch::tensor({1, 5, 8}, MooseTensor::intTensorOptions());   //  x dir;  x = 0
  _bottom = torch::tensor({2, 5, 6}, MooseTensor::intTensorOptions()); // y dir; y = 0

  //
  _reorder_indices = torch::tensor({6, 2, 5, 3, 0, 1, 7, 4, 8}, MooseTensor::intTensorOptions());
}
