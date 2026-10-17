/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MooseFunctionTensor.h"
#include "Function.h"
#include "MarlinUtils.h"
#include "TensorProblem.h"

registerMooseObject("MarlinApp", MooseFunctionTensor);

InputParameters
MooseFunctionTensor::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("Map a MooseFunction to a tensor.");
  params.addRequiredParam<FunctionName>("function", "Function to map.");
  // params.addParam<bool>("reciprocal", false, "Construct a reciprocal buffer");
  return params;
}

MooseFunctionTensor::MooseFunctionTensor(const InputParameters & parameters)
  : TensorOperator(parameters), FunctionInterface(this), _func(getFunction("function"))
{
}

void
MooseFunctionTensor::computeBuffer()
{
  auto buffer = torch::zeros(_tensor_problem.getShape(), torch::kDouble);

  // Use this rank's local grid size (not the global DomainAction::getGridSize()) and its
  // global index offset, so each rank evaluates the Function at its own correct physical
  // location under multi-rank real-space decomposition. getGridSize()/_n_global is only
  // correct here when running on a single rank, where it coincides with the local size.
  std::array<int64_t, 3> begin{{0, 0, 0}}, end{{0, 0, 0}};
  _domain.getLocalBounds(_domain.comm().rank(), begin, end);
  const auto & n = _domain.getLocalGridSize();
  const auto & dx = _domain.getGridSpacing();

  switch (_domain.getDim())
  {
    {
      case 1:
        auto b = buffer.accessor<double, 1>();
        for (const auto i : make_range(n[0]))
          b[i] = _func.value(_time, Point((begin[0] + i) * dx(0) + dx(0) / 2.0, 0.0, 0.0));
        break;
    }
    case 2:
    {
      auto b = buffer.accessor<double, 2>();
      for (const auto j : make_range(n[1]))
        for (const auto i : make_range(n[0]))
          b[i][j] = _func.value(_time,
                                Point((begin[0] + i) * dx(0) + dx(0) / 2.0,
                                      (begin[1] + j) * dx(1) + dx(1) / 2.0,
                                      0.0));
      break;
    }
    case 3:
    {
      auto b = buffer.accessor<double, 3>();
      for (const auto k : make_range(n[2]))
        for (const auto j : make_range(n[1]))
          for (const auto i : make_range(n[0]))
            b[i][j][k] = _func.value(_time,
                                     Point((begin[0] + i) * dx(0) + dx(0) / 2.0,
                                           (begin[1] + j) * dx(1) + dx(1) / 2.0,
                                           (begin[2] + k) * dx(2) + dx(2) / 2.0));
      break;
    }
    default:
      mooseError("Unsupported dimension");
  }

  _u = buffer.to(MooseTensor::floatTensorOptions());

  // In real-space mode, pad with ghost layers so this buffer's shape matches every other
  // real-space buffer's (TensorProblem::getLocalTensorShape()); mirrors ParsedCompute's
  // identical final padding step (ParsedCompute::computeBuffer()). The padded values are
  // placeholders only -- TensorProblem::exchangeGhostLayers() overwrites them (periodic
  // wraparound or a TensorBoundaryCondition's fillGhostCells()) before any ghosted read.
  if (_domain.isRealSpaceMode())
  {
    const auto desired = _tensor_problem.getLocalTensorShape({});
    std::vector<int64_t> pad;
    pad.reserve(_dim * 2);
    bool needs_pad = false;
    for (int d = static_cast<int>(_dim) - 1; d >= 0; --d)
    {
      const auto target = desired[d];
      const auto cur = n[d];
      const auto left_p = (target - cur) / 2;
      const auto right_p = target - cur - left_p;
      if (left_p > 0 || right_p > 0)
        needs_pad = true;
      pad.push_back(left_p);
      pad.push_back(right_p);
    }
    if (needs_pad)
      _u = torch::constant_pad_nd(_u, pad, 0.0);
  }
}

void
MooseFunctionTensor::realSpaceComputeBuffer()
{
  computeBuffer();
}
