/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMBoundaryCondition.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

InputParameters
LBMBoundaryCondition::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  MooseEnum boundary("top bottom left right front back wall");
  params.addRequiredParam<MooseEnum>(
      "boundary", boundary, "Edges/Faces where boundary condition is applied.");
  params.addClassDescription("LBMBoundaryCondition object.");
  return params;
}

LBMBoundaryCondition::LBMBoundaryCondition(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _grid_size(_lb_problem.getGridSize()),
    _boundary(getParam<MooseEnum>("boundary").getEnum<Boundary>())
{
  /**
   * Nodes that are adjacent to boundary will be set to 2, this will later be used in determining
   * the nodes for bounce-back
   * This will be achieved by shifting the mesh around in streaming directions and finding where
   * boundary hit happens
   */
  // if (_lb_problem.isBinaryMedia())
  // {
  //   auto new_mesh = _mesh.getBinaryMesh().clone();
  //   for (int64_t ic = 1; ic < _stencil._q; ic++)
  //   {
  //     int64_t ex = _stencil._ex[ic].item<int64_t>();
  //     int64_t ey = _stencil._ey[ic].item<int64_t>();
  //     int64_t ez = _stencil._ez[ic].item<int64_t>();
  //     torch::Tensor shifted_mesh = torch::roll(new_mesh, {ex, ey, ez}, {0, 1, 2});
  //     torch::Tensor adjacent_to_boundary = (shifted_mesh == 0) & (new_mesh == 1);
  //     new_mesh.masked_fill_(adjacent_to_boundary, 2);
  //   }
  //   // Deep copy new mesh
  //   // MooseTensor::printField(new_mesh, 1, 0);
  //   _mesh.setBinaryMesh(new_mesh);
  // }

  // if (_lb_problem.isBinaryMedia() && _domain.getDim() != 3)
  //   LBMBoundaryCondition::buildBoundaryIndices();
}

#if 0
int64_t
LBMBoundaryCondition::countNumberofBoundaries()
{
  /**
   * For efficiency, we count the number of boundaries first
   */

  LBMBoundaryCondition::determineBoundaryTypes();

  int64_t k = 0;
  int64_t num_of_boundaries = 0;
  for (int64_t i = 0; i < _shape_q[0]; i++)
    for (int64_t j = 0; j < _shape_q[1]; j++)
      for (int64_t ic = 0; ic < _shape_q[3]; ic++)
      {
        // Avoid calling item() repeatedly
        int64_t boundary_type = _boundary_types[i][j][k].item<int64_t>();

        if (boundary_type != -1)
        {
          int64_t if_stream_index = _all_boundary_types.size(0) * ic + boundary_type;

          // when streaming along ic is NOT possible at i, j, k i.e zero
          if (_if_stream[if_stream_index].item<int64_t>() == 0)
          {
            num_of_boundaries++;
          }
        }
      }

  return num_of_boundaries;
}

void
LBMBoundaryCondition::buildBoundaryIndices()
{
  /**
   * Building boundary indices
   */
  // const torch::Tensor & mesh_expanded =
  // _mesh.getBinaryMesh().unsqueeze(3).expand(expected_shape); auto mask = (mesh_expanded == 2) &
  // (_u == 0); _boundary_indices = torch::nonzero(mask); _boundary_indices =
  // _boundary_indices.to(MooseTensor::intTensorOptions());

  int64_t num_of_boundaries = countNumberofBoundaries();

  // initialize boundary indices
  _boundary_indices = torch::zeros({num_of_boundaries, 4}, MooseTensor::intTensorOptions());

  int64_t row_index = 0;
  int64_t k = 0;
  for (int64_t i = 0; i < _shape_q[0]; i++)
    for (int64_t j = 0; j < _shape_q[1]; j++)
      for (int64_t ic = 0; ic < _shape_q[3]; ic++)
      {
        // Avoid calling item() repeatedly
        int64_t boundary_type = _boundary_types[i][j][k].item<int64_t>();

        if (boundary_type != -1)
        {
          int64_t if_stream_index = _all_boundary_types.size(0) * ic + boundary_type;

          // when streaming along ic is NOT possible at i, j, k i.e zero
          if (_if_stream[if_stream_index].item<int64_t>() == 0)
          {
            _boundary_indices[row_index] = torch::tensor(
                {i, j, k, _stencil._op[ic].item<int64_t>()}, MooseTensor::intTensorOptions());
            row_index++;
          }
        }
      }
}

void
LBMBoundaryCondition::determineBoundaryTypes()
{
  /**
   * Scan the binary domain in a 3x3 window to determine boundary types
   * D2Q9 only
   */

  const torch::Tensor & binary_mesh = _lb_problem.getBinaryMedia();
  _boundary_types =
      torch::zeros({_shape[0], _shape[1], _shape[2]}, MooseTensor::intTensorOptions());

  _boundary_types.fill_(-1);

  // changes the order of D2Q9
  std::vector<int64_t> d2q9_order_to_new_order{7, 3, 6, 4, 0, 2, 8, 1, 5};

  int64_t k = 0;

  for (int64_t i = 0; i < _shape[0]; i++)
    for (int64_t j = 0; j < _shape[1]; j++)
    {
      if (binary_mesh[i][j][k].item<int64_t>() != 0)
      {
        // assembling binary string
        std::string string_of_binary_digits;
        // search in all directions
        for (int64_t ic = 0; ic < _stencil._q; ic++)
        {
          int64_t i_prime = i + _stencil._ex[d2q9_order_to_new_order[ic]].item<int64_t>();
          int64_t j_prime = j + _stencil._ey[d2q9_order_to_new_order[ic]].item<int64_t>();

          // ensuring periodicity
          i_prime = (i_prime < 0) ? i_prime + _shape[0] : i_prime;
          i_prime = (i_prime >= _shape[0]) ? i_prime - _shape[0] : i_prime;

          j_prime = (j_prime < 0) ? j_prime + _shape[1] : j_prime;
          j_prime = (j_prime >= _shape[1]) ? j_prime - _shape[1] : j_prime;

          //
          if (binary_mesh[i_prime][j_prime][k].item<int64_t>() == 0)
            string_of_binary_digits += '0';
          else
            string_of_binary_digits += '1';
        }

        // convert binary to decimal
        std::bitset<32> binary(string_of_binary_digits);
        int64_t decimal_number = binary.to_ulong();
        if (decimal_number != 511)
        {
          auto comparison_result = (_all_boundary_types == decimal_number);
          bool is_decimal_number_in_boundary_types = torch::any(comparison_result).item<bool>();

          if (!is_decimal_number_in_boundary_types)
          {
            std::string error_message =
                "Boundary type " + std::to_string(decimal_number) + " is not found";
            mooseError(error_message);
          }
          else
          {
            // the index where boundary type matches decimal_number
            int64_t index = torch::nonzero(comparison_result).item<int64_t>();

            _boundary_types[i][j][k] = index;
          }
        }
      }
    }
}
#endif

void
LBMBoundaryCondition::computeBuffer()
{
  switch (_boundary)
  {
    case Boundary::top:
      topBoundary();
      break;
    case Boundary::bottom:
      bottomBoundary();
      break;
    case Boundary::left:
      leftBoundary();
      break;
    case Boundary::right:
      rightBoundary();
      break;
    case Boundary::front:
      frontBoundary();
      break;
    case Boundary::back:
      backBoundary();
      break;
    case Boundary::wall:
      wallBoundary();
      break;
    default:
      mooseError("Undefined boundary names");
  }
  _lb_problem.maskedFillSolids(_u, 0);
}
