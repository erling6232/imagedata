import unittest
from imagedata.series import Series


class TestXNATTransport(unittest.TestCase):

    def test_something(self):
        si1 = Series('xnat://xnat.medtek.haukeland.no/NyreHaugesund/TK/fPACS_E00184/5')


if __name__ == '__main__':
    unittest.main()
