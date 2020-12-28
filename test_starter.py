from __future__ import unicode_literals
import unittest
import jetproj.fitting_master as fitting_master
import numpy as np
# import matplotlib.pyplot as plt
# import os
# import emcee

class StarterTest(unittest.TestCase):
    def test_timing(self):
        self.assertTrue(np.array_equal(fitting_master.open_time2(path='../'), fitting_master.open_time8('../')),
                        msg='epochs are not synchronized')

    def test_flux(self):
        self.assertTrue((False == (fitting_master.open_fl1(path='../') < 0)).any(), msg='you have a negative flux in fl1')
        self.assertTrue((False == (fitting_master.open_fl2(path='../') < 0)).any(), msg='you have a negative flux in fl2')


if __name__ == '__main__':
    unittest.main()
