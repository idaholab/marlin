/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMD3Q19.h"

registerMooseObject("MarlinApp", LBMD3Q19);

InputParameters
LBMD3Q19::validParams()
{
  InputParameters params = LatticeBoltzmannStencilBase::validParams();
  params.addClassDescription("LBMD3Q19 Stencil object.");
  return params;
}

LBMD3Q19::LBMD3Q19(const InputParameters & parameters) : LatticeBoltzmannStencilBase(parameters)
{
  _q = 19;
  // LBMD3Q19 lattice
  _ex = torch::tensor({0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1},
                      MooseTensor::intTensorOptions());
  _ey = torch::tensor({0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1},
                      MooseTensor::intTensorOptions());
  _ez = torch::tensor({0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0},
                      MooseTensor::intTensorOptions());
  _weights = torch::tensor(
      {
          1.0 / 3.0,  1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
          1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
          1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
      },
      MooseTensor::floatTensorOptions());

  _op = torch::tensor({0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15},
                      MooseTensor::intTensorOptions());
  // transformation matrix
  _M = torch::tensor(
      {{1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.},
       {-30., -11., -11., -11., -11., -11., -11., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.},
       {12., -4., -4., -4., -4., -4., -4., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.},
       {0., 1., -1., 0., 0., 0., 0., 1., -1., 1., -1., 1., -1., 1., -1., 0., 0., 0., 0.},
       {0., -4., 4., 0., 0., 0., 0., 1., -1., 1., -1., 1., -1., 1., -1., 0., 0., 0., 0.},
       {0., 0., 0., 1., -1., 0., 0., 1., 1., -1., -1., 0., 0., 0., 0., 1., -1., 1., -1.},
       {0., 0., 0., -4., 4., 0., 0., 1., 1., -1., -1., 0., 0., 0., 0., 1., -1., 1., -1.},
       {0., 0., 0., 0., 0., 1., -1., 0., 0., 0., 0., 1., 1., -1., -1., 1., 1., -1., -1.},
       {0., 0., 0., 0., 0., -4., 4., 0., 0., 0., 0., 1., 1., -1., -1., 1., 1., -1., -1.},
       {0., 2., 2., -1., -1., -1., -1., 1., 1., 1., 1., 1., 1., 1., 1., -2., -2., -2., -2.},
       {0., -4., -4., 2., 2., 2., 2., 1., 1., 1., 1., 1., 1., 1., 1., -2., -2., -2., -2.},
       {0., 0., 0., 1., 1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1., 0., 0., 0., 0.},
       {0., 0., 0., -2., -2., 2., 2., 1., 1., 1., 1., -1., -1., -1., -1., 0., 0., 0., 0.},
       {0., 0., 0., 0., 0., 0., 0., 1., -1., -1., 1., 0., 0., 0., 0., 0., 0., 0., 0.},
       {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -1., -1., 1.},
       {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -1., -1., 1., 0., 0., 0., 0.},
       {0., 0., 0., 0., 0., 0., 0., 1., -1., 1., -1., -1., 1., -1., 1., 0., 0., 0., 0.},
       {0., 0., 0., 0., 0., 0., 0., -1., -1., 1., 1., 0., 0., 0., 0., 1., -1., 1., -1.},
       {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., -1., -1., -1., -1., 1., 1.}},
      MooseTensor::floatTensorOptions());

  _M_inv = at::linalg_inv(_M);

  // relaxation matrix
  _S = torch::diag(torch::tensor({1. / 1.,
                                  1. / 1.19,
                                  1. / 1.4,
                                  1. / 1.4,
                                  1. / 1.0000,
                                  1. / 1.,
                                  1. / 1.0000,
                                  1. / 1.,
                                  1. / 1.0000,
                                  1. / 1.0000,
                                  1. / 1.4,
                                  1. / 1.000,
                                  1. / 1.4,
                                  1. / 1.000,
                                  1. / 1.000,
                                  1. / 1.000,
                                  1. / 1.98,
                                  1. / 1.98,
                                  1. / 1.98},
                                 MooseTensor::floatTensorOptions()));

  // indices where relaxation parameter related to kinematic viscosity (i.e. shear stress) is
  // located
  _id_kinematic_visc = torch::tensor({9, 11, 13, 14, 15}, MooseTensor::intTensorOptions());
  /**
   * incoming unknown distribution functions at every face
   * the opposite faces can be determined using _op vector
   * E.g. the opposite of _top[0] is _bottom[0] = _op[top[0]]
   */
  _left = torch::tensor({5, 11, 12, 15, 16}, MooseTensor::intTensorOptions()); // x dir; x = 0
  _right = _op.index({_left});                                                 // x dir; x = nx-1
  _bottom = torch::tensor({3, 7, 8, 15, 17}, MooseTensor::intTensorOptions()); // y dir; y = 0
  _top = _op.index({_bottom});                                                 // y dir; x = ny-1
  _front = torch::tensor({1, 7, 9, 11, 13}, MooseTensor::intTensorOptions());  // z dir ; z = 0
  _back = _op.index({_front});                                                 // z dir; z = nz-1

  _neutral_x = torch::tensor(
      {
          0,
          1,
          2,
          3,
          4,
          7,
          8,
          9,
          10,
      },
      MooseTensor::intTensorOptions());
}
