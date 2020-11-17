import unittest
import numpy as np
import os
import logging
import shutil
from simsopt.mhd.vmec_standalone import VmecStandalone, nested_lists_to_array
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
    
if __name__ == "__main__":
    unittest.main()
