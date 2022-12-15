import unittest

from .context import imagedata
from imagedata import Study
from pydicom.datadict import tag_for_keyword


class TestStudy(unittest.TestCase):

    def test_read(self):
        study = Study('data/dicom')

        for uid in study:
            series = study[uid]
            try:
                serdescr = series.seriesDescription
            except ValueError:
                serdescr = series.getDicomAttribute(tag_for_keyword('SequenceName'))
            self.assertIsNotNone(serdescr)
            self.assertIsNotNone(series.seriesNumber)
            self.assertIsNotNone(series.input_order)

if __name__ == '__main__':
    unittest.main()
