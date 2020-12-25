from __future__ import unicode_literals
import unittest
import jetproj.fitting_master as fitting_master
import numpy as np


class StarterTest(unittest.TestCase):
    def test_timing(self):
        self.assertTrue(np.array_equal(fitting_master.open_files()[4], fitting_master.open_files()[5]),
                        msg='epochs are not synchronized')

    def test_flux(self):
        self.assertTrue((False == (fitting_master.open_files()[2] < 0)).any(), msg='you have a negative flux in fl1')
        self.assertTrue((False == (fitting_master.open_files()[3] < 0)).any(), msg='you have a negative flux in fl2')


if __name__ == '__main__':
    unittest.main()
