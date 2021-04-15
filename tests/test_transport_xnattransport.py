import unittest
from imagedata.series import Series


class TestXNATTransport(unittest.TestCase):

    def test_something(self):
        si1 = Series('xnat://xnat.medtek.haukeland.no/NyreHaugesund/TK/fPACS_E00184/5')
        print(si1)
        print(si1.shape)


if __name__ == '__main__':
    unittest.main()
