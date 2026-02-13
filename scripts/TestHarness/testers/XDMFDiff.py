# *                    DO NOT MODIFY THIS HEADER
# *            Marlin, a Fourier spectral solver for MOOSE
# *
# *            Copyright 2024 Battelle Energy Alliance, LLC
# *                        ALL RIGHTS RESERVED
# *
# *        Licensed under LGPL 2.1, please see LICENSE for details
# *             https://www.gnu.org/licenses/lgpl-2.1.html

from FileTester import FileTester
from TestHarness import util
import os


class XDMFDiff(FileTester):

    @staticmethod
    def validParams():
        params = FileTester.validParams()
        params.addRequiredParam(
            "xdmfdiff", [], "A list of XDMF files to compare against the gold."
        )
        params.addParam("abs_error", None, "Absolute error threshold.")
        params.addParam("rel_error", None, "Relative error threshold.")
        params.addParam(
            "abs_zero", None, "Absolute floor for relative error denominator."
        )
        params.addParam("abs_tol", None, "Deprecated alias for abs_error.")
        params.addParam("rel_tol", None, "Deprecated alias for rel_error.")
        params.addParam(
            "center", "cell", "Attribute center(s) to compare: cell, node, or both."
        )
        params.addParam("step", None, "Optional timestep index to compare.")
        params.addParam("time", None, "Optional timestep time value to compare.")
        return params

    def __init__(self, name, params):
        FileTester.__init__(self, name, params)
        if self.specs["required_python_packages"] is None:
            self.specs["required_python_packages"] = "h5py numpy"
        elif "h5py" not in self.specs["required_python_packages"]:
            self.specs["required_python_packages"] += " h5py numpy"

    def getOutputFiles(self, options):
        return super().getOutputFiles(options) + self.specs["xdmfdiff"]

    def processResultsCommand(self, moose_dir, options):
        commands = []
        script = os.path.abspath(
            os.path.join(self.getMooseDir(), "..", "scripts", "xdmfdiff.py")
        )

        abs_error = self.specs["abs_error"] if self.specs.isValid("abs_error") else None
        if abs_error is None and self.specs.isValid("abs_tol"):
            abs_error = self.specs["abs_tol"]
        rel_error = self.specs["rel_error"] if self.specs.isValid("rel_error") else None
        if rel_error is None and self.specs.isValid("rel_tol"):
            rel_error = self.specs["rel_tol"]
        abs_zero = self.specs["abs_zero"] if self.specs.isValid("abs_zero") else None

        for file in self.specs["xdmfdiff"]:
            cmd = [script]
            gold = os.path.join(self.getTestDir(), self.specs["gold_dir"], file)
            test = os.path.join(self.getTestDir(), file)
            cmd.append(gold + " " + test)

            if abs_error is not None:
                cmd.append("--abs-error %s" % abs_error)
            if rel_error is not None:
                cmd.append("--rel-error %s" % rel_error)
            if abs_zero is not None:
                cmd.append("--abs-zero %s" % abs_zero)

            if self.specs.isValid("center"):
                cmd.append("--centers %s" % self.specs["center"])
            if self.specs.isValid("step") and self.specs["step"] is not None:
                cmd.append("--step %s" % self.specs["step"])
            if self.specs.isValid("time") and self.specs["time"] is not None:
                cmd.append("--time %s" % self.specs["time"])

            commands.append(" ".join(cmd))

        return commands

    def processResults(self, moose_dir, options, exit_code, runner_output):
        output = super().processResults(moose_dir, options, exit_code, runner_output)

        if self.isFail() or self.specs["skip_checks"]:
            return output

        if options.scaling and self.specs["scale_refine"]:
            return output

        for file in self.specs["xdmfdiff"]:
            if not os.path.exists(
                os.path.join(self.getTestDir(), self.specs["gold_dir"], file)
            ):
                output += "File Not Found: " + os.path.join(
                    self.getTestDir(), self.specs["gold_dir"], file
                )
                self.setStatus(self.fail, "MISSING GOLD FILE")
                break

        if not self.isFail():
            commands = self.processResultsCommand(moose_dir, options)

            for command in commands:
                exo_output = util.runCommand(command)
                output += "Running xdmfdiff: " + command + "\n" + exo_output
                if exo_output.startswith("ERROR:"):
                    self.setStatus(self.diff, "XDMFDIFF")
                    break

        return output
