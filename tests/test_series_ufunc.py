import unittest
import tempfile
import numpy as np

from src.imagedata.series import Series
import src.imagedata.axis as axis


class TestSeriesUfunc(unittest.TestCase):

    def test_not_implemented_median(self):
        si0 = Series(np.eye(4))
        b = np.median(si0)

    def test_min(self):
        si0 = Series('data/dicom/time/')
        a = si0.min()
        b = np.min(si0)

    def test_result_type(self):
        si0 = Series(np.eye(4))
        b = np.result_type(si0, 1.0)
        pass

    def test_rint(self):
        si0 = Series(np.eye(4))
        b = np.rint(si0)
        pass

    def test_concatenate_time(self):
        # Original series
        si0 = Series('data/dicom/time/')
        # Construct series with shifted time points
        si1 = Series('data/dicom/time/')
        time0 = si0.axes[0]
        dt = time0[1] - time0[0]
        t1 = time0[-1] + dt
        add_time = t1 - time0[0]
        si1.axes[0] = axis.VariableAxis(time0.name, si1.axes[0] + add_time)

        c = np.concatenate((si0, si1), axis=0)
        self.assertEqual(c.shape, (si0.shape[0] + si1.shape[0],
                                   si0.shape[1], si0.shape[2], si0.shape[3])
                         )
        tl = c.timeline
        tc = c.axes[0]
        self.assertEqual(6, len(tl))
        np.testing.assert_array_almost_equal(tc,
                                             np.concatenate((si0.axes[0].values, si1.axes[0]. values)))
        with tempfile.TemporaryDirectory() as d:
            c.write(d)



if __name__ == '__main__':
    unittest.main()
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    # runner = unittest.TextTestRunner(verbosity=2)
    # unittest.main(testRunner=runner)
