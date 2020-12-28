from __future__ import unicode_literals
import unittest
import jetproj.fitting_master as fitting_master
import numpy as np
import sys


class StarterTest(unittest.TestCase):
    def test_timing(self):
        np.testing.assert_array_equal(fitting_master.open_time2('../'), fitting_master.open_time8('../'))

    def test_flux(self):
        self.assertTrue(np.any(fitting_master.open_fl1(path='../') >= 0), msg='you have a negative flux in fl1')
        self.assertTrue(np.any(fitting_master.open_fl2(path='../') >= 0), msg='you have a negative flux in fl2')


if __name__ == '__main__':
    unittest.main()
