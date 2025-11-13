#*                    DO NOT MODIFY THIS HEADER
#*             Swift, a Fourier spectral solver for MOOSE
#*
#*            Copyright 2024 Battelle Energy Alliance, LLC
#*                        ALL RIGHTS RESERVED
#*
#*        Licensed under LGPL 2.1, please see LICENSE for details
#*             https://www.gnu.org/licenses/lgpl-2.1.html

import os, sys, subprocess, tempfile

import TestHarness
sys.path.append(os.path.join(TestHarness.__path__[0], 'tests'))
from TestHarnessTestCase import TestHarnessTestCase

#TEST_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..')

class SwiftTestHarnessTestCase(TestHarnessTestCase):
    """
    TestCase class for running TestHarness commands.
    """
