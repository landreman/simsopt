# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides an interface to the KNOSOS code by Velasco et
al.
"""

import os
import shutil
import logging
import subprocess
import numpy as np
from .._core.optimizable import Optimizable
from .._core.util import ObjectiveFailure

logger = logging.getLogger(__name__)


class Knosos(Optimizable):
    """
    If you want to run KNOSOS from a Boozer object, then provide that
    object as the boozer argument. If instead you want to run KNOSOS
    for a fixed netcdf file, you can specify that boozmn.nc file in
    input_files, and set boozer=None.

    If mpi is not None, then only proc0_groups will run KNOSOS.

    Do not include input.surfaces in the list of input files, since
    this file will be generated automatically.
    """

    def __init__(self,
                 boozer=None,
                 s=0.5,
                 input_files=["input.fastions", "input.model", "input.parameters"],
                 exe="xknosos",
                 mpi=None):

        self.boozer = boozer
        self.exe = exe
        self.need_to_run_code = True
        self.mpi = mpi
        self.counter = -1

        # Record all the input files:
        self.input_files = [os.path.abspath(filename) for filename in input_files]
        logger.debug(f"input_files: {self.input_files}")

        # Force s to be a list:
        try:
            ss = list(s)
        except:
            ss = [s]

        self.s = sorted(ss)

        if self.boozer is None:
            self.depends_on = []
        else:
            self.depends_on = ["boozer"]
            # knosos also requires an adjacent surface. To get this, we
            # first need to get the final ns from vmec.
            ns_array = boozer.equil.indata.ns_array
            ns = 0
            for ns_j in ns_array:
                if ns_j > 0:
                    ns = ns_j
            ds = 1.0 / (ns - 1)
            for s_j in self.s:
                boozer.register({s_j, s_j - ds})

    def get_dofs(self):
        return np.array([])

    def set_dofs(self, x):
        self.need_to_run_code = True

    def run(self):
        """
        Actually run KNOSOS.
        """
        if (self.mpi is not None) and (not self.mpi.proc0_groups):
            logger.debug("This proc is skipping Knosos.run since it is not a group leader.")
            return

        if not self.need_to_run_code:
            logger.debug("No need to re-run KNOSOS")
            return

        logger.debug("About to run KNOSOS")
        self.counter += 1

        if self.boozer is not None:
            self.boozer.run()

        if self.mpi is None:
            group = ""
        else:
            group = "_{:03}".format(self.mpi.group)
        dirname = "knosos{}_{:06}".format(group, self.counter)
        os.mkdir(dirname)
        os.chdir(dirname)

        # Write all the input files needed by knosos:

        if self.boozer is not None:
            self.boozer.bx.write_boozmn("boozmn.nc")

        for filename in self.input_files:
            shutil.copy(filename, '.')

        f = open("input.surfaces", "w")
        f.write("&surfaces\n")
        f.write(f"NS={len(self.s)}\n")
        f.write("S=")
        for s_j in self.s:
            f.write(f" {s_j}")
        f.write("\n/\n")
        f.close()

        # Call the KNOSOS executable:
        knosos_command = self.exe.split()
        logger.debug(f"About to call subprocess.run with {knosos_command}")
        subprocess.run(self.exe.split())

        # Read output:
        ns = len(self.s)
        filename = "fort.6700"
        try:
            data = np.loadtxt(filename)
        except:
            os.chdir("..")
            raise ObjectiveFailure("Unable to read KNOSOS output file " \
                                   + filename + " in dir " + dirname)
        # Ensure data is a 2D array:
        if data.ndim == 1:
            self.data = data.reshape((1, -1))
        else:
            self.data = data

        os.chdir("..")
        logger.info("Successfully ran KNOSOS and read output")
        self.need_to_run_code = False

    def Gamma_c(self, js=0):
        """
        Return the Gamma_alpha quantity to estimate fast particle confinement.
        """
        self.run()
        return self.data[js, 0]

    def Gamma_alpha(self, js=0):
        """
        Return the Gamma_alpha quantity to estimate fast particle confinement.
        """
        self.run()
        return self.data[js, 3]
