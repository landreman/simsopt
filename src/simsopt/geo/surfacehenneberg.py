import numpy as np
import logging

from .surface import Surface
from .surfacerzfourier import SurfaceRZFourier

logger = logging.getLogger(__name__)


class SurfaceHenneberg(Surface):
    """
    This class represents a toroidal surface using the
    parameterization in Henneberg, Helander, and Drevlak,
    arXiv:2105.00768 (2021).

    In the implementation here, stellarator symmetry is assumed.

    The continuous degrees of freedom are :math:`\{rho_{m,n}, b_n,
    R_{0,n}^H, Z_{0,n}^H\}`.  These variables correspond to the
    attributes ``rhomn``, ``bn``, ``R0nH``, and ``Z0nH`` respectively,
    which are all numpy arrays.  There is also a discrete degree of
    freedom :math:`\alpha` which should be 0.5 or -0.5. The attribute
    ``alpha_times_2`` corresponds to :math:`2\alpha`.
    """

    def __init__(self, nfp, alpha_times_2, mmax, nmax):
        if alpha_times_2 != 1 and alpha_times_2 != -1:
            raise ValueError('alpha_times_2 must be 1 or -1')
        
        self.nfp = nfp
        self.alpha_times_2 = alpha_times_2
        self.mmin = mmin
        self.mmax = mmax
        self.nmin = nmin
        self.nmax = nmax
        self.stellsym = True
        self.allocate()
        self.recalculate = True

        # Initialize to an axisymmetric torus with major radius 1m and
        # minor radius 0.1m
        self.set_Delta(1, 0, 1.0)
        self.set_Delta(0, 0, 0.1)
        Surface.__init__(self)

    def __repr__(self):
        return "SurfaceHenneberg " + str(hex(id(self))) + " (nfp=" + \
            str(self.nfp) + ", alpha=" + str(self.alpha_times_2 * 0.5) \
            + ", mmax=" + str(self.mmax) + ", nmax=" + str(self.nmax) + ")"

    def allocate(self):
        """
        Create the arrays for the continuous degrees of freedom. Also set
        the names of the dofs.
        """
        logger.debug("Allocating SurfaceHenneberg")
        # Note that for simpicity, the Z0nH array contains an element
        # for n=0 even though this element is always 0. Similarly, the
        # rhomn array has some elements for (m=0, n<0) even though
        # these elements are always zero.
        
        self.R0nH = np.zeros(self.mmax + 1)
        self.Z0nH = np.zeros(self.mmax + 1)
        self.b0nH = np.zeros(self.mmax + 1)
        
        self.ndim = 2 * self.nmax + 1
        myshape = (self.mmax + 1, self.ndim)
        self.rhomn = np.zeros(myshape)
        
        self.names = []
        for n in range(self.nmax + 2):
            self.names.append('R0nH(' + str(n) + ')')
        for n in range(1, self.nmax + 2):
            self.names.append('Z0nH(' + str(n) + ')')
        for n in range(self.nmax + 2):
            self.names.append('bn(' + str(n) + ')')
        # Handle m = 0 modes in rho_mn:
        for n in range(self.nmax + 2):
            self.names.append('rhomn(0,' + str(n) + ')')
        # Handle m > 0 modes in rho_mn:
        for m in range(1, self.mmax + 1):
            for n in range(-self.nmax, self.nmax + 1):
                self.names.append('rhomn(' + str(m) + ',' + str(n) + ')')

    def get_Delta(self, m, n):
        """
        Return a particular :math:`\Delta_{m,n}` coefficient.
        """
        return self.Delta[m - self.mmin, n - self.nmin]

    def set_Delta(self, m, n, val):
        """
        Set a particular :math:`\Delta_{m,n}` coefficient.
        """
        self.Delta[m - self.mmin, n - self.nmin] = val
        self.recalculate = True
        self.recalculate_derivs = True

    def get_dofs(self):
        """
        Return a 1D numpy array with all the degrees of freedom.
        """
        num_dofs = (self.mmax - self.mmin + 1) * (self.nmax - self.nmin + 1)
        return np.reshape(self.Delta, (num_dofs,), order='F')

    def set_dofs(self, v):
        """
        Set the shape coefficients from a 1D list/array
        """

        n = len(self.get_dofs())
        if len(v) != n:
            raise ValueError('Input vector should have ' + str(n) + \
                             ' elements but instead has ' + str(len(v)))

        # Check whether any elements actually change:
        if np.all(np.abs(self.get_dofs() - np.array(v)) == 0):
            logger.info('set_dofs called, but no dofs actually changed')
            return

        logger.info('set_dofs called, and at least one dof changed')
        self.recalculate = True
        self.recalculate_derivs = True

        self.Delta = v.reshape((self.mmax - self.mmin + 1, self.nmax - self.nmin + 1), order='F')

    def fixed_range(self, mmin, mmax, nmin, nmax, fixed=True):
        """
        Set the 'fixed' property for a range of m and n values.

        All modes with m in the interval [mmin, mmax] and n in the
        interval [nmin, nmax] will have their fixed property set to
        the value of the 'fixed' parameter. Note that mmax and nmax
        are included (unlike the upper bound in python's range(min,
        max).)
        """
        for m in range(mmin, mmax + 1):
            for n in range(nmin, nmax + 1):
                self.set_fixed('Delta({},{})'.format(m, n), fixed)

    def to_RZFourier(self):
        """
        Return a SurfaceRZFourier object with the identical shape. This
        routine implements eq (4.5)-(4.6) in the Henneberg paper, plus
        m=0 terms for R0 and Z0.
        """
        mpol = self.mmax
        ntor = self.nmax + 1 # More modes are needed in the SurfaceRZFourier because some indices are shifted by +/- 2*alpha.
        s = SurfaceRZFourier(nfp=self.nfp, stellsym=True, mpol=mpol, ntor=ntor)
        
        # Set Rmn.
        # Handle the 1d arrays (R0nH, bn):
        for nprime in range(self.nmax + 1):
            n = nprime
            # Handle the R0nH term:
            s.set_rc(0, n, self.R0nH[n])
            # Handle the b_n term:
            s.set_rc(1, n, 0.25 * self.bn[nprime])
            # Handle the b_{-n} term:
            n = -nprime
            s.set_rc(1, n, s.get_rc(1, n) + 0.25 * self.bn[nprime])
            # Handle the b_{n-2alpha} term:
            n = nprime + self.alpha_times_2
            s.set_rc(1, n, s.get_rc(1, n) - 0.25 * self.bn[nprime])
            # Handle the b_{-n+2alpha} term:
            n = -nprime + self.alpha_times_2
            s.set_rc(1, n, s.get_rc(1, n) - 0.25 * self.bn[nprime])
        # Handle the 2D rho terms:
        for m in range(self.mmax + 1):
            for nprime in range(-self.nmax, self.nmax + 1):
                # Handle the rho_{m, -n} term:
                n = -nprime
                s.set_rc(m, n, s.get_rc(m, n) + 0.5 * self.get_rhomn(m, nprime))
                # Handle the rho_{m, -n+2alpha} term:
                n = -nprime + self.alpha_times_2
                s.set_rc(m, n, s.get_rc(m, n) + 0.5 * self.get_rhomn(m, nprime))
        
        # Set Zmn.
        # Handle the 1d arrays (Z0nH, bn):
        for nprime in range(self.nmax + 1):
            n = nprime
            # Handle the Z0nH term:
            s.set_zs(0, n, -self.Z0nH[n])
            # Handle the b_n term:
            s.set_zs(1, n, 0.25 * self.bn[nprime])
            # Handle the b_{-n} term:
            n = -nprime
            s.set_zs(1, n, s.get_zs(1, n) + 0.25 * self.bn[nprime])
            # Handle the b_{n-2alpha} term:
            n = nprime + self.alpha_times_2
            s.set_zs(1, n, s.get_zs(1, n) + 0.25 * self.bn[nprime])
            # Handle the b_{-n+2alpha} term:
            n = -nprime + self.alpha_times_2
            s.set_zs(1, n, s.get_zs(1, n) + 0.25 * self.bn[nprime])
        # Handle the 2D rho terms:
        for m in range(self.mmax + 1):
            for nprime in range(-self.nmax, self.nmax + 1):
                # Handle the rho_{m, -n} term:
                n = -nprime
                s.set_zs(m, n, s.get_zs(m, n) + 0.5 * self.get_rhomn(m, nprime))
                # Handle the rho_{m, -n+2alpha} term:
                n = -nprime + self.alpha_times_2
                s.set_zs(m, n, s.get_zs(m, n) - 0.5 * self.get_rhomn(m, nprime))
        
        return s

    def area_volume(self):
        """
        Compute the surface area and the volume enclosed by the surface.
        """
        if self.recalculate:
            logger.info('Running calculation of area and volume')
        else:
            logger.info('area_volume called, but no need to recalculate')
            return

        self.recalculate = False

        # Delegate to the area and volume calculations of SurfaceRZFourier():
        s = self.to_RZFourier()
        self._area = s.area()
        self._volume = s.volume()

    def area(self):
        """
        Return the area of the surface.
        """
        self.area_volume()
        return self._area

    def volume(self):
        """
        Return the volume of the surface.
        """
        self.area_volume()
        return self._volume
