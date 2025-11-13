/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseApp.h"
#include "MarlinUtils.h"

namespace MooseTensor
{
std::string torchDevice();
std::string precision();
}

class DomainAction;
class MarlinInit;

class MarlinApp : public MooseApp
{
public:
  static InputParameters validParams();

  MarlinApp(const InputParameters & parameters);
  virtual ~MarlinApp();

  static void registerApps();
  static void registerAll(Factory & f, ActionFactory & af, Syntax & s);

  /// called from the ComputeDevice action
  void setTorchDevice(std::string device, const MooseTensor::Key<DomainAction> &);
  /// called from the unit test app
  static void setTorchDeviceStatic(std::string device, const MooseTensor::Key<MarlinInit> &);
  /// called from the Domain action
  void setTorchPrecision(std::string precision, const MooseTensor::Key<DomainAction> &);
};
