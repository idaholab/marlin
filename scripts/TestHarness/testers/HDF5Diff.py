#*                    DO NOT MODIFY THIS HEADER
#*            Marlin, a Fourier spectral solver for MOOSE
#*
#*            Copyright 2024 Battelle Energy Alliance, LLC
#*                        ALL RIGHTS RESERVED
#*
#*        Licensed under LGPL 2.1, please see LICENSE for details
#*             https://www.gnu.org/licenses/lgpl-2.1.html

from FileTester import FileTester
import os
import sys

class HDF5Diff(FileTester):

    @staticmethod
    def validParams():
        params = FileTester.validParams()
        params.addRequiredParam('hdf5diff', [], 'A list of files to compare against the gold.')
        params.addParam('abs_tol', 1e-15, 'Absolute tolerance.')
        return params

    def __init__(self, name, params):
        FileTester.__init__(self, name, params)
        if self.specs['required_python_packages'] is None:
             self.specs['required_python_packages'] = 'h5py numpy'
        elif 'h5py' not in self.specs['required_python_packages']:
            self.specs['required_python_packages'] += ' h5py numpy'

    def getOutputFiles(self, options):
        return self.specs['hdf5diff']

    def processResults(self, moose_dir, options, exit_code, runner_output):
        """
        Perform hdf5 diff
        """
        import h5py
        import numpy as np

        # Call base class processResults
        output = super().processResults(moose_dir, options, exit_code, runner_output)
        if self.isFail():
            return output

        abs_tol = self.specs['abs_tol']

        # Loop through files
        specs = self.specs
        for filename in specs['hdf5diff']:

            # Error if gold file does not exist
            if not os.path.exists(os.path.join(self.getTestDir(), specs['gold_dir'], filename)):
                output += "File Not Found: " + os.path.join(self.getTestDir(), specs['gold_dir'], filename)
                self.setStatus(self.fail, 'MISSING GOLD FILE')
                break

            # Perform diff
            else:
                gold = os.path.join(self.getTestDir(), specs['gold_dir'], filename)
                test = os.path.join(self.getTestDir(), filename)

                gold_file = h5py.File(gold)
                test_file = h5py.File(test)

                # check available datasets
                datasets = gold_file.keys()
                if datasets != test_file.keys():
                    self.setStatus(self.fail, 'MISMATCHING DATASETS')
                    return f"Datasets in gold file:\n{list(datasets)}\nDatasets in test file:\n{list(test_file.keys())}\n"

                # compare all sets
                for dataset in datasets:
                    gold_set = gold_file[dataset][...]
                    test_set = test_file[dataset][...]

                    # check shape
                    if gold_set.shape != test_set.shape:
                        output += f"Mismatching shape for dataset '{dataset}' (gold:{gold_set.shape}, test:{test_set.shape})\n"
                        self.setStatus(self.fail, 'HDF5 DIFF')

                    else:
                        diff = np.max(np.abs(gold_set - test_set))
                        if diff > abs_tol:
                            output += f"Absolute tolerance exceeded in '{dataset}' (diff:{diff}, abs_tol:{abs_tol})\n"
                            self.setStatus(self.fail, 'HDF5 DIFF')

        return output
