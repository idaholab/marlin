/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/
#include "MarlinTestApp.h"
#include "MarlinApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "MooseSyntax.h"

InputParameters
MarlinTestApp::validParams()
{
  InputParameters params = MarlinApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  return params;
}

MarlinTestApp::MarlinTestApp(InputParameters parameters) : MooseApp(parameters)
{
  MarlinTestApp::registerAll(
      _factory, _action_factory, _syntax, getParam<bool>("allow_test_objects"));
}

MarlinTestApp::~MarlinTestApp() {}

void
MarlinTestApp::registerAll(Factory & f, ActionFactory & af, Syntax & s, bool use_test_objs)
{
  MarlinApp::registerAll(f, af, s);
  if (use_test_objs)
  {
    Registry::registerObjectsTo(f, {"MarlinTestApp"});
    Registry::registerActionsTo(af, {"MarlinTestApp"});
  }
}

void
MarlinTestApp::registerApps()
{
  registerApp(MarlinApp);
  registerApp(MarlinTestApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
// External entry point for dynamic application loading
extern "C" void
MarlinTestApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  MarlinTestApp::registerAll(f, af, s);
}
extern "C" void
MarlinTestApp__registerApps()
{
  MarlinTestApp::registerApps();
}
