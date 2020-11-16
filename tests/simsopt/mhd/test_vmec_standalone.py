import unittest
import numpy as np
import os
from simsopt.mhd.vmec_standalone import VmecStandalone
from . import TEST_DIR

class VmecStandaloneTests(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
