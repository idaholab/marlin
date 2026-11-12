/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LatticeBoltzmannStencilBase.h"

namespace
{
template <typename T>
std::vector<T>
toHost(const torch::Tensor & t)
{
  if (!t.defined())
    return {};

  const auto c = t.to(torch::kCPU).to(c10::CppTypeToScalarType<T>::value).contiguous();
  const auto * data = c.template data_ptr<T>();
  return std::vector<T>(data, data + c.numel());
}
}

InputParameters
LatticeBoltzmannStencilBase::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.addClassDescription("LB Stencil object.");
  params.registerBase("LBStencil");
  return params;
}

LatticeBoltzmannStencilBase::LatticeBoltzmannStencilBase(const InputParameters & parameters)
  : MooseObject(parameters)
{
}

const LatticeBoltzmannStencilBase::HostTables &
LatticeBoltzmannStencilBase::host() const
{
  if (!_host)
    _host = HostTables{toHost<int64_t>(_ex),
                       toHost<int64_t>(_ey),
                       toHost<int64_t>(_ez),
                       toHost<int64_t>(_op),
                       toHost<int64_t>(_left),
                       toHost<int64_t>(_right),
                       toHost<int64_t>(_bottom),
                       toHost<int64_t>(_top),
                       toHost<int64_t>(_front),
                       toHost<int64_t>(_back),
                       toHost<Real>(_weights)};
  return *_host;
}
