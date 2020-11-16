# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class that handles the VMEC equilibrium code.
"""

import logging
import os.path
import numpy as np
import f90nml

from simsopt.core import Optimizable, SurfaceRZFourier

logger = logging.getLogger(__name__)

class VmecStandalone(Optimizable):
    """
    This class represents the VMEC equilibrium code.
    """
    def __init__(self, filename=None, exe="xvmec2000"):
        """
        Constructor
        """
        if filename is None:
            # Read default input file, which should be in the same
            # directory as this file:
            filename = os.path.join(os.path.dirname(__file__), 'input.default')
            logger.info("Initializing a VMEC object from defaults in " \
                            + filename)
        else:
            logger.info("Initializing a VMEC object from file: " + filename)

        self.read_input(filename)
        self.depends_on = ["boundary"]
        self.fixed = np.full(len(self.get_dofs()), True)
        self.names = ['delt', 'tcon0', 'phiedge', 'curtor', 'gamma']

    def _parse_namelist(self, nml, varname, default):
        """
        For each variable that might be in the namelist, get the value
        from the namelist if a value is present, otherwise use the
        default.
        """
        if varname in nml:
            setattr(self, varname, nml[varname])
        else:
            setattr(self, varname, default)
        
    def read_input(self, filename):
        nmls = f90nml.read(filename)
        # We only care about the 'indata' namelist
        nml = nmls['indata']
        self._parse_namelist(nml, 'delt', 0.7)
        self._parse_namelist(nml, 'tcon0', 1.0)
        self._parse_namelist(nml, 'phiedge', 1.0)
        self._parse_namelist(nml, 'curtor', 0.0)
        self._parse_namelist(nml, 'gamma', 0.0)
        self._parse_namelist(nml, 'mpol', 1)
        self._parse_namelist(nml, 'ntor', 0)
        self._parse_namelist(nml, 'nfp', 1)
        self._parse_namelist(nml, 'lasym', False)

        # We can assume rbc and zbs are specified in the namelist.
        rc = np.array(nml['rbc'], dtype='f')
        zs = np.array(nml['zbs'], dtype='f')
        # When converting the list of lists from f90nml to a numpy
        # array, unspecified values become NaN. The next 2 lines fix
        # this.
        rc[np.isnan(rc)] = 0
        zs[np.isnan(zs)] = 0

        rbc_first_n = nml.start_index['rbc'][0]
        rbc_last_n = rbc_first_n + rc.shape[1] - 1
        zbs_first_n = nml.start_index['zbs'][0]
        zbs_last_n = zbs_first_n + rc.shape[1] - 1
        ntor_boundary = np.max(np.abs(np.array([rbc_first_n, rbc_last_n, zbs_first_n, zbs_last_n], dtype='i')))

        rbc_first_m = nml.start_index['rbc'][1]
        rbc_last_m = rbc_first_m + rc.shape[0] - 1
        zbs_first_m = nml.start_index['zbs'][1]
        zbs_last_m = zbs_first_m + rc.shape[0] - 1
        mpol_boundary = np.max((rbc_last_m, zbs_last_m))
        logger.debug('Input file has ntor_boundary={} mpol_boundary={}'.format(ntor_boundary, mpol_boundary))
        self.boundary = SurfaceRZFourier(nfp=self.nfp, stelsym=not self.lasym,
                                         mpol=mpol_boundary, ntor=ntor_boundary)
        
        # Transfer boundary shape data from the namelist to the surface object:
        for jm in range(rc.shape[0]):
            m = jm + nml.start_index['rbc'][1]
            for jn in range(rc.shape[1]):
                n = jn + nml.start_index['rbc'][0]
                self.boundary.set_rc(m, n, rc[jm, jn])
                
        for jm in range(zs.shape[0]):
            m = jm + nml.start_index['zbs'][1]
            for jn in range(zs.shape[1]):
                n = jn + nml.start_index['zbs'][0]
                self.boundary.set_zs(m, n, zs[jm, jn])
                
        self.need_to_run_code = True
        
    def get_dofs(self):
        return np.array([self.delt, self.tcon0, self.phiedge, self.curtor, self.gamma])

    def set_dofs(self, x):
        self.need_to_run_code = True
        self.delt = x[0]
        self.tcon0 = x[1]
        self.phiedge = x[2]
        self.curtor = x[3]
        self.gamma = x[4]
    
