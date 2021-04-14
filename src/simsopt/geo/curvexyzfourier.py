from .curve import Curve
import simsgeopp as sgpp
import numpy as np
from jax.ops import index, index_add
import jax.numpy as jnp
from math import pi
from .curve import JaxCurve


class CurveXYZFourier(sgpp.CurveXYZFourier, Curve):

    r"""
       CurveXYZFourier is a curve that is represented in cartesian
       coordinates using the following Fourier series:

           x(phi) = \sum_{m=0}^{order} x_{c,m}cos(m*phi) + \sum_{m=1}^order x_{s,m}sin(m*phi)
           y(phi) = \sum_{m=0}^{order} y_{c,m}cos(m*phi) + \sum_{m=1}^order y_{s,m}sin(m*phi)
           z(phi) = \sum_{m=0}^{order} z_{c,m}cos(m*phi) + \sum_{m=1}^order z_{s,m}sin(m*phi)

       The dofs are stored in the order

           [x_{c,0},...,x_{c,order},x_{s,1},...,x_{s,order},y_{c,0},....]
    """

    def __init__(self, quadpoints, order):
        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1, quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        sgpp.CurveXYZFourier.__init__(self, quadpoints, order)
        Curve.__init__(self)

    def get_dofs(self):
        return np.asarray(sgpp.CurveXYZFourier.get_dofs(self))

    def set_dofs(self, dofs):
        sgpp.CurveXYZFourier.set_dofs(self, dofs)
        for d in self.dependencies:
            d.invalidate_cache()

    @staticmethod
    def load_curves_from_file(filename, order=None, ppp=20, delimiter=','):
        """
        This function loads a file containing Fourier coefficients for several coils.
        The file is expected to have 6 * num_coils many columns, and order+1 many rows.
        The columns are in the following order,

            sin_x_coil1, cos_x_coil1, sin_y_coil1, cos_y_coil1, sin_z_coil1, cos_z_coil1, sin_x_coil2, cos_x_coil2, sin_y_coil2, cos_y_coil2, sin_z_coil2, cos_z_coil2,  ...

        """
        coil_data = np.loadtxt(filename, delimiter=delimiter)

        assert coil_data.shape[1] % 6 == 0
        assert order <= coil_data.shape[0]-1

        num_coils = coil_data.shape[1]//6
        coils = [CurveXYZFourier(order*ppp, order) for i in range(num_coils)]
        for ic in range(num_coils):
            dofs = coils[ic].dofs
            dofs[0][0] = coil_data[0, 6*ic + 1]
            dofs[1][0] = coil_data[0, 6*ic + 3]
            dofs[2][0] = coil_data[0, 6*ic + 5]
            for io in range(0, min(order, coil_data.shape[0]-1)):
                dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
                dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
                dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
                dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
                dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
                dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
            coils[ic].set_dofs(np.concatenate(dofs))
        return coils


def jaxfouriercurve_pure(dofs, quadpoints, order):
    k = len(dofs)//3
    coeffs = [dofs[:k], dofs[k:(2*k)], dofs[(2*k):]]
    points = quadpoints
    gamma = np.zeros((len(points), 3))
    for i in range(3):
        gamma = index_add(gamma, index[:, i], coeffs[i][0])
        for j in range(1, order+1):
            gamma = index_add(gamma, index[:, i], coeffs[i][2*j-1] * jnp.sin(2*pi*j*points))
            gamma = index_add(gamma, index[:, i], coeffs[i][2*j] * jnp.cos(2*pi*j*points))
    return gamma


class JaxCurveXYZFourier(JaxCurve):

    """
    A Python+Jax implementation of the CurveXYZFourier class.  There is
    actually no reason why one should use this over the C++ implementation in
    simsgeopp, but the point of this class is to illustrate how jax can be used
    to define a geometric object class and calculate all the derivatives (both
    with respect to dofs and with respect to the angle phi) automatically.
    """

    def __init__(self, quadpoints, order):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        pure = lambda dofs, points: jaxfouriercurve_pure(dofs, points, order)
        self.order = order
        self.coefficients = [np.zeros((2*order+1,)), np.zeros((2*order+1,)), np.zeros((2*order+1,))]
        super().__init__(quadpoints, pure)

    def num_dofs(self):
        return 3*(2*self.order+1)

    def get_dofs(self):
        return np.concatenate(self.coefficients)

    def set_dofs_impl(self, dofs):
        counter = 0
        for i in range(3):
            self.coefficients[i][0] = dofs[counter]
            counter += 1
            for j in range(1, self.order+1):
                self.coefficients[i][2*j-1] = dofs[counter]
                counter += 1
                self.coefficients[i][2*j] = dofs[counter]
                counter += 1
