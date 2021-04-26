import unittest
from imagedata.series import Series
from imagedata.transports import Transport


class TestXNATTransport(unittest.TestCase):

    # @unittest.skip("skipping test_something")
    def test_transport(self):
        # si1 = Series('xnat://xnat.server/project/subject/experiment/scan')
        # si1.seriesNumber = 9999
        # si1.seriesDescription = 'tttt'
        # si1.write('xnat://xnat.server/project/subject/experiment')
        pass


if __name__ == '__main__':
    unittest.main()
