/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ProjectVectorTensorAux.h"
#include "DomainAction.h"
#include "TensorProblem.h"
#include "MarlinTypes.h"

#if 0
registerMooseObject("MarlinApp", ProjectVectorTensorAux);

InputParameters
ProjectVectorTensorAux::validParams()
{
  InputParameters params = ArrayAuxKernel::validParams();
  params.addClassDescription("Project a vectorial Tensor buffer onto an auxiliary variable");
  params.addRequiredParam<TensorInputBufferName>("buffer", "The buffer to read from");
  return params;
}

ProjectVectorTensorAux::ProjectVectorTensorAux(const InputParameters & parameters)
  : ArrayAuxKernel(parameters),
    TensorProblemInterface(this),
    DomainInterface(this),
    _cpu_buffer(_tensor_problem.getRawCPUBuffer(getParam<TensorInputBufferName>("buffer"))),
    _dim(_domain.getDim()),
    _n(_domain.getGridSize()),
    _grid_spacing(_domain.getGridSpacing())
{
  mooseWarning("ProjectVectorTensorAux is under development and not tested.");
}

RealEigenVector
ProjectVectorTensorAux::computeValue()
{
  auto getElement = [this](const int64_t & component)
  {
    const Point shift(_grid_spacing(0) / 2.0, _grid_spacing(1) / 2.0, _grid_spacing(2) / 2.0);
    Point p = isNodal() ? (*_current_node + shift) : _current_elem->centroid();

    using at::indexing::TensorIndex;
    switch (_dim)
    {
      case 1:
        return _cpu_buffer.index(
            {TensorIndex(int64_t(p(0) / _grid_spacing(0)) % _n[0]), TensorIndex(component)});

      case 2:
        return _cpu_buffer.index({TensorIndex(int64_t(p(0) / _grid_spacing(0)) % _n[0]),
                                  TensorIndex(int64_t(p(1) / _grid_spacing(1)) % _n[1]),
                                  TensorIndex(component)});

      case 3:
        return _cpu_buffer.index({TensorIndex(int64_t(p(0) / _grid_spacing(0)) % _n[0]),
                                  TensorIndex(int64_t(p(1) / _grid_spacing(1)) % _n[1]),
                                  TensorIndex(int64_t(p(2) / _grid_spacing(2)) % _n[2]),
                                  TensorIndex(component)});
    }

    mooseError("Internal error (invalid dimension)");
  };

  RealEigenVector v(_var.count());

  for (unsigned int i = 0; i < _var.count(); ++i)
  {
    const auto element = getElement(int64_t(i));
    if (_cpu_buffer.dtype() == torch::kFloat32)
      v(i) = element.item<float>();
    else if (_cpu_buffer.dtype() == torch::kFloat64)
      v(i) = element.item<double>();
    else
      mooseError("Unsupported output type");
  }
  return v;
}

#endif
