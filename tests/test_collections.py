import tempfile
import unittest

from .context import imagedata
from imagedata import Study, Patient, Cohort


class TestCollections(unittest.TestCase):

    def test_read_study(self):
        # study = Study('data/dicom')
        # study = Study('data/dicom/cohort.zip?p2/20221220-094921.932000')
        study = Study('data/dicom/cohort.zip?cohort/IMAGEDATA_P2.MR.ERLING_IMAGEDATA*')

        for uid in study:
            series = study[uid]
            try:
                seriesDescription = series.seriesDescription
            except ValueError:
                seriesDescription = series.getDicomAttribute('SequenceName')
            self.assertIsNotNone(seriesDescription)
            self.assertIsNotNone(series.seriesNumber)
            self.assertIsNotNone(series.input_order)
        self.assertIsNotNone(study.studyInstanceUID)

    def test_write_study(self):
        # study = Study('data/dicom')
        study = Study('data/dicom/cohort.zip?cohort/IMAGEDATA_P2.MR.ERLING_IMAGEDATA*')
        with tempfile.TemporaryDirectory() as d:
            study.write(d)

    def test_read_patient(self):
        patient = Patient('data/dicom/cohort.zip?cohort/IMAGEDATA_P2.MR.ERLING_IMAGEDATA*')

        for uid in patient:
            study = patient[uid]
            studyDescription = study.studyDescription
            self.assertIsNotNone(studyDescription)
            self.assertIsNotNone(study.studyDate)
            self.assertIsNotNone(study.studyID)
        self.assertIsNotNone(patient.patientID)

    def test_write_patient(self):
        patient = Patient('data/dicom/cohort.zip?cohort/IMAGEDATA_P2.MR.ERLING_IMAGEDATA*')
        with tempfile.TemporaryDirectory() as d:
            patient.write(d)

    def test_read_cohort(self):
        cohort = Cohort('data/dicom/cohort.zip')

        for patientID in cohort:
            patient = cohort[patientID]
            cohort_description = cohort.data
            self.assertIsNotNone(cohort_description)
            self.assertIsNotNone(patient.patientName)
            self.assertIsNotNone(patient.patientID)

    def test_write_cohort(self):
        cohort = Cohort('data/dicom/cohort.zip')
        with tempfile.TemporaryDirectory() as d:
            cohort.write(d)


if __name__ == '__main__':
    unittest.main()
