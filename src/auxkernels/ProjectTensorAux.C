/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ProjectTensorAux.h"
#include "DomainAction.h"
#include "TensorProblem.h"
#include "MarlinTypes.h"

registerMooseObject("MarlinApp", ProjectTensorAux);

InputParameters
ProjectTensorAux::validParams()
{
  InputParameters params = AuxKernel::validParams();
  params.addClassDescription("Project a Tensor buffer onto an auxiliary variable");
  params.addRequiredParam<TensorInputBufferName>("buffer", "The buffer to read from");
  return params;
}

ProjectTensorAux::ProjectTensorAux(const InputParameters & parameters)
  : AuxKernel(parameters),
    TensorProblemInterface(this),
    DomainInterface(this),
    _cpu_buffer(_tensor_problem.getRawCPUBuffer(getParam<TensorInputBufferName>("buffer"))),
    _dim(_domain.getDim()),
    _n(_domain.getGridSize()),
    _grid_spacing(_domain.getGridSpacing())
{
}

Real
ProjectTensorAux::computeValue()
{
  auto getElement = [this]()
  {
    const Point shift(_grid_spacing(0) / 2.0, _grid_spacing(1) / 2.0, _grid_spacing(2) / 2.0);
    Point p = isNodal() ? (*_current_node + shift) : _current_elem->vertex_average();

    using at::indexing::TensorIndex;
    switch (_dim)
    {
      case 1:
        return _cpu_buffer.index({TensorIndex(int64_t(p(0) / _grid_spacing(0)) % _n[0])});

      case 2:
        return _cpu_buffer.index({TensorIndex(int64_t(p(0) / _grid_spacing(0)) % _n[0]),
                                  TensorIndex(int64_t(p(1) / _grid_spacing(1)) % _n[1])});

      case 3:
        return _cpu_buffer.index({TensorIndex(int64_t(p(0) / _grid_spacing(0)) % _n[0]),
                                  TensorIndex(int64_t(p(1) / _grid_spacing(1)) % _n[1]),
                                  TensorIndex(int64_t(p(2) / _grid_spacing(2)) % _n[2])});
    }

    mooseError("Internal error (invalid dimension)");
  };

  const auto element = getElement();

  if (_cpu_buffer.dtype() == torch::kFloat32)
    return element.item<float>();
  else if (_cpu_buffer.dtype() == torch::kFloat64)
    return element.item<double>();
  else
    mooseError("Unsupported output type");
}
