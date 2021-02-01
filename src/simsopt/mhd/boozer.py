# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class that handles the transformation to
Boozer coordinates, and an optimization target for quasisymmetry.
"""

import logging
import os.path
from typing import Union, Iterable
import numpy as np

booz_xform_found = True
try:
    import booz_xform
except:
    booz_xform_found = False

from simsopt.core import Optimizable
from simsopt.mhd import Vmec

logger = logging.getLogger(__name__)

# This next function can be deleted I think.
def closest_index(grid: Iterable[float], val: float) -> int:
    """
    Given a grid of values, find the grid point that is closest to an
    abitrary value.

    Args:
        grid: A list of values.
        val: We will return the index of the closest grid point to this value.
    """
    return np.argmin(np.abs(grid - val))


class Boozer(Optimizable):
    """
    This class handles the transformation to Boozer coordinates.

    A Boozer instance maintains a set "s", which is a registry of the
    surfaces on which other objects want Boozer-coordinate data. When
    the run() method is called, the Boozer transformation is carried
    out on all these surfaces. The registry can be cleared at any time
    by setting the s attribute to {}.
    """
    def __init__(self,
                 equil: Vmec,
                 mpol: int = None,
                 ntor: int = None) -> None:
        """
        Constructor
        """
        if not booz_xform_found:
            raise RuntimeError("To use a Boozer object, the booz_xform package"
                               "must be installed. Run 'pip install -v booz_xform'")
        self.equil = equil
        self.depends_on = ["equil"]
        self.mpol = mpol
        self.ntor = ntor
        self.bx = booz_xform.Booz_xform()
        self.s = set()
        self.need_to_run_code = True

    def register(self, s: Union[float, Iterable[float]]) -> None:
        """
        This function is called by objects that depend on this Boozer
        object, to indicate that they will want Boozer data on the
        given set of surfaces.

        Args:
            s: 1 or more surfaces on which Boozer data will be requested.
        """
        # Force input surface data to be a set:
        try:
            ss = set(s)
        except:
            ss = {s}

        for new_s in ss:
            if new_s < 0 or new_s > 1:
                raise ValueError("Normalized toroidal flux values s must lie"
                                 "in the interval [0, 1]")
        logger.info("Adding entries to Boozer registry: {}".format(ss))
        self.s = self.s.union(ss)
        self.need_to_run_code = True

    def run(self):
        """
        Run booz_xform on all the surfaces that have been registered.
        """
        if not self.need_to_run_code:
            logger.info("Boozer.run() called but no need to re-run Boozer transformation.")
            return
        s = sorted(list(self.s))
        logger.info("Preparing to run Boozer transformation. Registry:{}".format(s))

        if isinstance(self.equil, Vmec):
            self.equil.run()
            wout = self.equil.VMEC.wout # Shorthand

            # Get the half-grid points that are closest to the requested values
            ns = wout.ns
            s_full = np.linspace(0, 1, ns)
            ds = s_full[1] - s_full[0]
            s_half = s_full[1:] - 0.5 * ds

            compute_surfs = []
            self.s_to_index = dict()
            for ss in s:
                index = np.argmin(np.abs(s_half - ss))
                compute_surfs.append(index)
                self.s_to_index[ss] = index
                
            # Eliminate any duplicates
            compute_surfs = sorted(list(set(compute_surfs)))
            logger.info("compute_surfs={}".format(compute_surfs))

            # Transfer data in memory from VMEC to booz_xform
            self.bx.asym = bool(wout.lasym)
            self.bx.nfp = wout.nfp

            self.bx.mpol = wout.mpol
            self.bx.ntor = wout.ntor
            self.bx.mnmax = wout.mnmax
            self.bx.xm = wout.xm
            self.bx.xn = wout.xn
            
            self.bx.mpol_nyq = wout.xm_nyq[-1]
            self.bx.ntor_nyq = wout.xn_nyq[-1] / nfp
            self.bx.mnmax_nyq = wout.mnmax_nyq
            self.bx.xm_nyq = wout.xm_nyq
            self.bx.xn_nyq = wout.xn_nyq

            if wout.lasym:
                rmns = wout.rmns
                zmnc = wout.zmnc
                lmnc = wout.lmnc
                bmns = wout.bmns
                bsubumns = wout.bsubumns
                bsubvmns = wout.bsubvmns
            else:
                # For stellarator-symmetric configs, the asymmetric
                # arrays have not been initialized.
                arr = np.array([[]])
                rmns = arr
                zmnc = arr
                lmnc = arr
                bmns = arr
                bsubumns = arr
                bsubvmns = arr
                
            # For quantities that depend on radius, booz_xform handles
            # interpolation and discarding the rows of zeros:
            self.bx.init_from_vmec(wout.ns,
                                   wout.iotas,
                                   wout.rmnc,
                                   rmns,
                                   zmnc,
                                   wout.zmns,
                                   lmnc,
                                   wout.lmns,
                                   wout.bmnc,
                                   bmns,
                                   wout.bsubumnc,
                                   bsubumns,
                                   wout.bsubvmnc,
                                   bsubvmns)
            self.bx.compute_surfs = compute_surfs

        else:
            # Cases for SPEC, GVEC, etc could be added here.
            raise ValueError("equil is not an equilibrium type supported by"
                             "Boozer")
        
        logger.info("About to call booz_xform.Booz_xform.run().")
        self.bx.run()
        logger.info("Returned from calling booz_xform.Booz_xform.run().")
        self.need_to_run_code = False
        
        
class Quasisymmetry(Optimizable):
    """
    This class is used to compute the departure from quasisymmetry on
    a given flux surface based on the Boozer spectrum.
    """
    def __init__(self,
                 boozer: Boozer,
                 s: Union[float, Iterable[float]],
                 m: int,
                 n: int,
                 normalization: str = "B00",
                 weight: str = "even") -> None:
        """
        Constructor

        Args:
            boozer: A Boozer object on which the calculation will be based.
            s: The normalized toroidal magnetic flux for the flux surface to analyze. Should be in the range [0, 1].
            m: The departure from symmetry B(m * theta - nfp * n * zeta) will be reported.
            n: The departure from symmetry B(m * theta - nfp * n * zeta) will be reported.
            normalization: A uniform normalization applied to all bmnc harmonics.
            weight: An option for a m- or n-dependent weight to be applied to the bmnc amplitudes.
        """
        self.boozer = boozer
        self.m = m
        self.n = n
        self.normalization = normalization
        self.weight = weight
        self.depends_on = ['boozer']

        # If s is not already iterable, make it so:
        try:
            iter(s)
        except:
            s = [s]
        self.s = s
        boozer.register(s)


    def J(self) -> Iterable:
        """
        Carry out the calculation of the quasisymmetry error.
        """
        # The next line is the expensive part of the calculation:
        self.boozer.run()
        
        symmetry_error = []
        for js, s in enumerate(self.s):
            index = self.boozer.s_to_index[s]
            bmnc = self.boozer.bx.bmnc_b[:, index]
            xm = self.boozer.bx.xm_b
            xn = self.boozer.bx.xn_b / self.boozer.bx.nfp

            if self.m != 0 and self.m != 1:
                raise ValueError("m for quasisymmetry should be 0 or 1.")

            # Find the indices of the symmetric modes:
            if self.n == 0:
                # Quasi-axisymmetry
                symmetric = (xn == 0)

            elif self.m == 0:
                # Quasi-poloidal symmetry
                symmetric = (xm == 0)

            else:
                # Quasi-helical symmetry
                symmetric = (xm * self.n + xn * self.m == 0)
                # Stellopt takes the "and" of this with mod(xm, self.m),
                # which does not seem necessary since self.m must be 1 to
                # get here.
            nonsymmetric = np.logical_not(symmetric)

            # Scale all bmnc modes so the average |B| is 1 or close to 1:

            if self.normalization == "B00":
                # Normalize by the (m,n) = (0,0) mode amplitude:
                assert xm[0] == 0
                assert xn[0] == 0
                bnorm = bmnc[0]

            elif self.normalization == "symmetric":
                # Normalize by sqrt(sum_{symmetric modes} B{m,n}^2)
                temp = bmnc[symmetric]
                bnorm = np.sqrt(np.dot(temp, temp))

            else:
                raise ValueError("Unrecognized value for normalization in Quasisymmetry")

            bmnc /= bnorm

            # Apply any weight that depends on m and/or n:

            if self.weight == "even":
                # Evenly weight each bmnc mode. Normalize by the m=n=0 mode on that surface.
                symmetry_error.append(bmnc[nonsymmetric])

            elif self.weight == "stellopt":
                # Stellopt applies a m-dependent radial weight:
                s_used = self.s # This line may need to be changed
                rad_sigma = np.full_like(xm, s_used * s_used)
                rad_sigma[xm < 3] = s_used
                rad_sigma[xm == 3] = s_used ** 1.5
                temp = bmnc / rad_sigma
                symmetry_error.append(temp[nonsymmetric])

            else:
                raise ValueError("Unrecognized value for weight in Quasisymmetry")

        return np.array(symmetry_error).flatten()
