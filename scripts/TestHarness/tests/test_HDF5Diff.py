#*                    DO NOT MODIFY THIS HEADER
#*            Marlin, a Fourier spectral solver for MOOSE
#*
#*            Copyright 2024 Battelle Energy Alliance, LLC
#*                        ALL RIGHTS RESERVED
#*
#*        Licensed under LGPL 2.1, please see LICENSE for details
#*             https://www.gnu.org/licenses/lgpl-2.1.html

import subprocess
from MarlinTestHarnessTestCase import MarlinTestHarnessTestCase

class TestHarnessTester(MarlinTestHarnessTestCase):
    def testShapeMismatch(self):
        """
        Test for error due to shape mismatch in HDF5Diff
        """
        with self.assertRaises(subprocess.CalledProcessError) as cm:
            self.runTests('-i', 'hdf5diffs', '--re', 'shape_mismatch')

        e = cm.exception
        self.assertRegex(e.output, r"tester\.shape_mismatch:.*Mismatching shape for dataset 'c\.0' \(gold:\(5, 6\), test:\(9, 9\)\)")

    def testValueMismatch(self):
        """
        Test for error due to value mismatch in HDF5Diff
        """
        with self.assertRaises(subprocess.CalledProcessError) as cm:
            self.runTests('-i', 'hdf5diffs', '--re', 'value_mismatch')

        e = cm.exception
        self.assertRegex(e.output, r"tester\.value_mismatch:.*Absolute tolerance exceeded in 'c\.0' \(diff:0\.4268.*, abs_tol:1e-15\)")

    def testDataSetMismatch(self):
        """
        Test for error due to dataset mismatch in HDF5Diff
        """
        with self.assertRaises(subprocess.CalledProcessError) as cm:
            self.runTests('-i', 'hdf5diffs', '--re', 'dataset_mismatch')

        e = cm.exception
        self.assertRegex(e.output, r"tester\.dataset_mismatch:.*\['c\.0', 'c\.1', 'c\.2', 'mu\.0', 'mu\.1', 'mu\.2'\]")
