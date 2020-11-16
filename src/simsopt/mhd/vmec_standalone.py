# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class that handles the VMEC equilibrium code.
"""

import logging
import os.path
import datetime
import numpy as np
import f90nml

from simsopt.core import Optimizable, SurfaceRZFourier

logger = logging.getLogger(__name__)

def bool2str(b):
    """ Convert a python bool to a "T" or "F" for use in fortran namelists """
    return "T" if b else "F"

def nested_lists_to_array(ll):
    """
    Convert a ragged list of lists to a 2D numpy array.  Any entries
    that are None are replaced by 0.
    """
    mdim = len(ll)
    ndim = np.max([len(x) for x in ll])
    arr = np.zeros((mdim, ndim))
    for jm, l in enumerate(ll):
        for jn, x in enumerate(l):
            if x is not None:
                arr[jm, jn] = x
    return arr

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
            
    def _to_list(self, varname):
        """
        Make sure 1D arrays are actually arrays rather than ints or floats
        """
        var = getattr(self, varname)
        try:
            setattr(self, varname, list(var))
        except:
            setattr(self, varname, [var])

    def read_input(self, filename):
        nmls = f90nml.read(filename)
        # We only care about the 'indata' namelist
        nml = nmls['indata']
        self._parse_namelist(nml, 'delt', 0.7)
        self._parse_namelist(nml, 'niter', 10000)
        self._parse_namelist(nml, 'nstep', 200)
        self._parse_namelist(nml, 'tcon0', 1.0)
        self._parse_namelist(nml, 'ns_array', [13])
        self._parse_namelist(nml, 'niter_array', [3000])
        self._parse_namelist(nml, 'ftol_array', [1e-10])
        self._parse_namelist(nml, 'precon_type', 'none')
        self._parse_namelist(nml, 'prec2d_threshold', 1e-19)
        self._parse_namelist(nml, 'lasym', False)
        self._parse_namelist(nml, 'nfp', 1)
        self._parse_namelist(nml, 'mpol', 1)
        self._parse_namelist(nml, 'ntor', 0)
        self._parse_namelist(nml, 'ntheta', None)
        self._parse_namelist(nml, 'nzeta', None)
        self._parse_namelist(nml, 'phiedge', 1.0)
        self._parse_namelist(nml, 'lfreeb', False)
        self._parse_namelist(nml, 'mgrid_file', '')
        self._parse_namelist(nml, 'nvacskip', 6)
        self._parse_namelist(nml, 'gamma', 0.0)
        self._parse_namelist(nml, 'bloat', 1.0)
        self._parse_namelist(nml, 'spres_ped', 1.0)
        self._parse_namelist(nml, 'pres_scale', 1.0)
        self._parse_namelist(nml, 'pmass_type', 'power_series')
        self._parse_namelist(nml, 'am', [0.0])
        self._parse_namelist(nml, 'curtor', 0.0)
        self._parse_namelist(nml, 'ncurr', 1)
        self._parse_namelist(nml, 'ac', [0.0])
        self._parse_namelist(nml, 'ai', [0.0])
        self._parse_namelist(nml, 'piota_type', 'power_series')
        self._parse_namelist(nml, 'pcurr_type', 'power_peries')
        #self._parse_namelist(nml, '', )

        # Make sure 1D arrays are actually arrays rather than ints or floats:
        self._to_list('ns_array')
        self._to_list('ftol_array')
        self._to_list('niter_array')
        self._to_list('am')
                 
        # We can assume rbc and zbs are specified in the namelist.
        # f90nml returns rbc and zbs as a list of lists where the
        # inner lists do not necessarily all have the same
        # dimension. Hence we need to be careful when converting to
        # numpy arrays.
        rc = nested_lists_to_array(nml['rbc'])
        zs = nested_lists_to_array(nml['zbs'])

        """
        rc = np.array(nml['rbc'], dtype='f')
        zs = np.array(nml['zbs'], dtype='f')
        # When converting the list of lists from f90nml to a numpy
        # array, unspecified values become NaN. The next 2 lines fix
        # this.
        rc[np.isnan(rc)] = 0
        zs[np.isnan(zs)] = 0
        """
        
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

    def write_input(self, filename):
        """
        Write a VMEC input file.
        """

        f = open(filename, 'w')
        f.write('! Input file generated by simsopt on {}\n\n'\
                .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write('&INDATA\n')
        f.write('DELT = {}\n'.format(self.delt))
        f.write('NITER = {}\n'.format(self.niter))
        f.write('NSTEP = {}\n'.format(self.nstep))
        f.write('TCON0 = {}\n'.format(self.tcon0))
        f.write('NS_ARRAY    =')
        [f.write('{:8} '.format(x)) for x in self.ns_array]
        f.write('\nFTOL_ARRAY  =')
        [f.write('{:8} '.format(x)) for x in self.ftol_array]
        f.write('\nNITER_ARRAY =')
        [f.write('{:8} '.format(x)) for x in self.niter_array]
        f.write('\nPRECON_TYPE = "{}"\n'.format(self.precon_type))
        f.write('PREC2D_THRESHOLD = {}\n'.format(self.prec2d_threshold))
        f.write('LASYM = {}\n'.format(bool2str(self.lasym)))
        f.write('NFP = {}\n'.format(self.nfp))
        f.write('MPOL = {}\n'.format(self.mpol))
        f.write('NTOR = {}\n'.format(self.ntor))
        f.write('PHIEDGE = {:.15e}\n'.format(self.phiedge))
        f.write('LFREEB = {}\n'.format(bool2str(self.lfreeb)))
        f.write('MGRID_FILE = "{}"\n'.format(self.mgrid_file))
        if self.ntheta is not None:
            f.write('NTHETA = {}\n'.format(self.ntheta))
        if self.nzeta is not None:
            f.write('NZETA = {}\n'.format(self.nzeta))
        f.write('NVACSKIP = {}\n'.format(self.nvacskip))
        f.write('GAMMA = {}\n'.format(self.gamma))
        f.write('BLOAT = {}\n'.format(self.bloat))
        f.write('SPRES_PED = {}\n'.format(self.spres_ped))
        f.write('PRES_SCALE = {}\n'.format(self.pres_scale))
        f.write('PMASS_TYPE = "{}"\n'.format(self.pmass_type))
        f.write('AM = ')
        [f.write('{:.15e} '.format(x)) for x in self.am]
        f.write('\nCURTOR = {:.15e}\n'.format(self.curtor))
        f.write('NCURR = {}\n'.format(self.ncurr))
        f.write('PIOTA_TYPE = "{}"\n'.format(self.piota_type))
        f.write('PCURR_TYPE = "{}"\n'.format(self.pcurr_type))
        f.write('AI = ')
        [f.write('{:.15e} '.format(x)) for x in self.ai]
        f.write('\nAC = ')
        [f.write('{:.15e} '.format(x)) for x in self.ac]
        f.write('\n! The magnetic axis shape is set to something singular to '
                'force VMEC to recalculate a good guess for it that depends '
                'only on the boundary shape.\n')
        f.write('RAXIS_CC = 0\n')
        f.write('ZAXIS_CS = 0\n')
        #f.write('={}\n'.format(self.))
        #f.write('={}\n'.format(self.))

        # Convert boundary to RZFourier if needed:
        boundary_RZFourier = self.boundary.to_RZFourier()
        # VMEC does not allow mpol or ntor above 101:
        mpol_capped = np.min((boundary_RZFourier.mpol, 101))
        ntor_capped = np.min((boundary_RZFourier.ntor, 101))
        for m in range(mpol_capped + 1):
            for n in range(-ntor_capped, ntor_capped + 1):
                rc = boundary_RZFourier.get_rc(m, n)
                zs = boundary_RZFourier.get_zs(m, n)
                if (rc != 0) or (zs != 0):
                    f.write('RBC({:3},{:3}) = {:22.15e}   ZBS({:3},{:3}) = {:22.15e}\n'\
                            .format(n, m, rc, n, m, zs))
        f.write('/\n')
        f.close()
        
    def get_dofs(self):
        return np.array([self.delt, self.tcon0, self.phiedge, self.curtor, self.gamma])

    def set_dofs(self, x):
        self.need_to_run_code = True
        self.delt = x[0]
        self.tcon0 = x[1]
        self.phiedge = x[2]
        self.curtor = x[3]
        self.gamma = x[4]
    
