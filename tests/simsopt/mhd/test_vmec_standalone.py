import unittest
import numpy as np
import os
import logging
import shutil
from simsopt.mhd.vmec_standalone import VmecStandalone, nested_lists_to_array
from simsopt.core.least_squares_problem import LeastSquaresProblem
from simsopt.core.serial_solve import least_squares_serial_solve
from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)

# See if a vmec executable can be found. If not, we will skip certain
# tests below.
exe = shutil.which('xvmec2000')
if exe is None:
    print('Trying to find xvmec2000')
    # Locations of xvmec2000 on Matt's laptop and the Github actions CI:
    try_exes = ['/Users/mattland/stellopt_github/develop/STELLOPT/bin/xvmec2000',
               '/home/runner/work/simsopt/simsopt/STELLOPT/VMEC2000/Release/xvmec2000']
    for try_exe in try_exes:
        if os.path.isfile(try_exe):
            exe = try_exe
vmec_standalone_found = (exe is not None)
logger.info("vmec standalone executable: {}".format(exe))


class VmecStandaloneTests(unittest.TestCase):
    def test_init_defaults(self):
        """
        Just create a Vmec instance using the standard constructor,
        and make sure we can read some of the attributes.
        """
        v = VmecStandalone()
        self.assertEqual(v.nfp, 5)
        self.assertFalse(v.lasym)
        self.assertEqual(v.mpol, 5)
        self.assertEqual(v.ntor, 4)
        self.assertAlmostEqual(v.delt, 0.5)
        self.assertAlmostEqual(v.tcon0, 2.0)
        self.assertAlmostEqual(v.phiedge, 1.0)
        self.assertAlmostEqual(v.curtor, 0.0)
        self.assertAlmostEqual(v.gamma, 0.0)
        self.assertEqual(v.ncurr, 1)
        self.assertFalse(v.lfreeb)
        self.assertTrue(v.need_to_run_code)

    def test_nested_lists_to_array(self):
        """
        Test the utility function used to convert the rbc and zbs data
        from f90nml to a 2D numpy array.
        """
        list_of_lists = [[42]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[42]])
        np.testing.assert_allclose(arr1, arr2)

        list_of_lists = [[42], [1, 2, 3]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[42, 0, 0],
                         [ 1, 2, 3]])
        np.testing.assert_allclose(arr1, arr2)

        list_of_lists = [[None, 42], [1, 2, 3]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[0, 42, 0],
                         [1,  2, 3]])
        np.testing.assert_allclose(arr1, arr2)

        list_of_lists = [[42, 43, 44], [1, 2, 3]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[42, 43, 44],
                         [ 1,  2,  3]])
        np.testing.assert_allclose(arr1, arr2)

        list_of_lists = [[42, 43, 44, 45], [1, 2, 3]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[42, 43, 44, 45],
                         [ 1,  2,  3,  0]])
        np.testing.assert_allclose(arr1, arr2)

        
    def test_init_from_file(self):
        """
        Try creating a Vmec instance from a specified input file.
        """

        filename = os.path.join(TEST_DIR, 'input.li383_low_res')

        v = VmecStandalone(filename)
        self.assertEqual(v.nfp, 3)
        self.assertEqual(v.mpol, 4)
        self.assertEqual(v.ntor, 3)
        self.assertEqual(v.boundary.mpol, 6)
        self.assertEqual(v.boundary.ntor, 4)

        places = 5
        
        # n = 0, m = 0:
        self.assertAlmostEqual(v.boundary.get_rc(0, 0), 1.3782, places=places)

        # n = 0, m = 1:
        self.assertAlmostEqual(v.boundary.get_zs(1, 0), 4.6465E-01, places=places)

        # n = 1, m = 1:
        self.assertAlmostEqual(v.boundary.get_zs(1, 1), 1.6516E-01, places=places)

        # n = -4, m = 6:
        self.assertAlmostEqual(v.boundary.get_rc(6, -4), 3.4247E-06, places=places)

        # n = 4, m = 6:
        self.assertAlmostEqual(v.boundary.get_zs(6, 4), 3.4674E-04, places=places)

        #self.assertEqual(v.ncurr, 1)
        #self.assertFalse(v.free_boundary)
        self.assertTrue(v.need_to_run_code)


    @unittest.skipIf(not vmec_standalone_found, "VMEC standalone executable not found")
    def test_run(self):
        """
        Try running VMEC and reading in the output.
        """
        filename = os.path.join(TEST_DIR, 'input.li383_low_res')

        v = VmecStandalone(filename, exe=exe)
        v.run()
        
        self.assertAlmostEqual(v.wout.betatotal, \
                                   0.0426215030653306, places=4)
        
        self.assertAlmostEqual(v.wout.iotaf[-1], \
                                   0.655517502443946, places=3)

        self.assertAlmostEqual(v.wout.rmnc[0, 0], \
                                   1.4783578816835392, places=3)
    
    @unittest.skipIf(not vmec_standalone_found, "VMEC standalone executable not found")
    def test_integrated_standalone_vmec_stellopt_scenarios_1dof(self):
        """
        This script implements the "1DOF_circularCrossSection_varyR0_targetVolume"
        example from
        https://github.com/landreman/stellopt_scenarios

        This optimization problem has one independent variable, representing
        the mean major radius. The problem also has one objective: the plasma
        volume. There is not actually any need to run an equilibrium code like
        VMEC since the objective function can be computed directly from the
        boundary shape. But this problem is a fast way to test the
        optimization infrastructure with VMEC.

        Details of the optimum and a plot of the objective function landscape
        can be found here:
        https://github.com/landreman/stellopt_scenarios/tree/master/1DOF_circularCrossSection_varyR0_targetVolume
        """
        for grad in [True, False]:
            # Start with a default surface, which is axisymmetric with major
            # radius 1 and minor radius 0.1.
            equil = VmecStandalone(exe=exe)
            surf = equil.boundary

            # Set the initial boundary shape. Here is one way to do it:
            surf.set('rc(0,0)', 1.0)
            # Here is another syntax that works:
            surf.set_rc(0, 1, 0.1)
            surf.set_zs(0, 1, 0.1)

            surf.set_rc(1, 0, 0.1)
            surf.set_zs(1, 0, 0.1)
            equil.ntor = 0

            # VMEC parameters are all fixed by default, while surface
            # parameters are all non-fixed by default. You can choose
            # which parameters are optimized by setting their 'fixed'
            # attributes.
            surf.all_fixed()
            surf.set_fixed('rc(0,0)', False)

            # Each Target is then equipped with a shift and weight, to become a
            # term in a least-squares objective function
            desired_volume = 0.15
            term1 = (equil.volume, desired_volume, 1)

            # A list of terms are combined to form a nonlinear-least-squares
            # problem.
            prob = LeastSquaresProblem([term1])

            # Check that the problem was set up correctly:
            self.assertEqual(len(prob.dofs.names), 1)
            self.assertEqual(prob.dofs.names[0][:7], 'rc(0,0)')
            np.testing.assert_allclose(prob.x, [1.0])
            self.assertEqual(prob.dofs.all_owners, [equil, surf])
            self.assertEqual(prob.dofs.dof_owners, [surf])

            # Solve the minimization problem:
            least_squares_serial_solve(prob, grad=grad)

            """
            print("At the optimum,")
            print(" rc(m=0,n=0) = ", surf.get_rc(0, 0))
            print(" volume, according to VMEC    = ", equil.volume())
            print(" volume, according to Surface = ", surf.volume())
            print(" objective function = ", prob.objective())
            """
            
            self.assertAlmostEqual(surf.get_rc(0, 0), 0.7599088773175, places=5)
            self.assertAlmostEqual(equil.volume(), 0.15, places=6)
            self.assertAlmostEqual(surf.volume(), 0.15, places=6)
            self.assertLess(np.abs(prob.objective()), 1.0e-15)

if __name__ == "__main__":
    unittest.main()
