/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

class ActionWarehouse;
class DomainAction;
class MooseBase;

class DomainInterface
{
public:
  DomainInterface(MooseBase * moose_base);

protected:
  const DomainAction & _domain;

private:
  const DomainAction & getDomain(MooseBase * moose_base);
};
