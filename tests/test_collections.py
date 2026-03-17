import os.path
import tempfile
import unittest
from numpy.random import default_rng

from imagedata import Series, Study, Patient, Cohort
from imagedata.formats import UnknownInputError


class TestStudy(unittest.TestCase):

    def test_read_study(self):
        # study = Study('data/dicom')
        # study = Study('data/dicom/cohort.zip?p2/20221220-094921.932000')
        study = Study('data/dicom/cohort.zip?cohort/P2/S1')

        for uid in study:
            series = study[uid]
            self.assertEqual('dicom', series.input_format)
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
        study = Study('data/dicom/cohort.zip?cohort/P2/S1')
        with tempfile.TemporaryDirectory() as d:
            study.write(d)

    def test_kwargs(self):
        def _read_study():
            si2 = Study(
                'data/dicom/time/time00/Image_00020.dcm',
                input_format='dicom',
                input_echo=2)

        si1 = Study(
            'data/dicom/time/time00/Image_00020.dcm',
            input_format='dicom',
            input_echo=1)
        self.assertRaises(UnknownInputError, _read_study)

    def test_indexed_dict(self):
        si1 = Study(
            'data/dicom/time/time00',
            input_format='dicom'
        )
        s = si1[0]
        self.assertIsInstance(s, Series, 'Study with key:0 does not return a Series object')

    def test_two_acqnum(self):
        si = Series(
            # 'data/dicom/time/time00/Image_00020.dcm',
            'data/dicom/time/time00',
            input_format = 'dicom'
        )
        with tempfile.TemporaryDirectory() as d:
            si.setDicomAttribute('AcquisitionNumber', 1)
            si.write(os.path.join(d, '1'))
            si.setDicomAttribute('AcquisitionNumber', 2)
            si.setDicomAttribute('SliceThickness', 4)
            si.write(os.path.join(d, '2'), opts={'keep_uid': True})
            study = Study(d, input_format='dicom', opts={'split_acquisitions': True})

    def test_anonymize_dicom_study(self):
        study = Study('data/dicom/cohort.zip?cohort/P2/S1')
        anon_study = study.anonymize()
        with tempfile.TemporaryDirectory() as d:
            anon_study.write(d, keep_uid=True)
            study1 = Study(d)
            self.assertEqual(len(study), len(study1))
            for uid in range(len(study)):
                self.assertEqual(study[uid].shape, study1[uid].shape)

    def test_anonymize_non_dicom_study(self):
        rng = default_rng()
        series1 = Series(rng.standard_normal(24).reshape((2,3,4))*100, dtype=int)
        series2 = Series(rng.standard_normal(210).reshape((5,6,7))*100, dtype=int)
        series2.studyInstanceUID = series1.studyInstanceUID
        study = Study({'1': series1, '2': series2})
        study.studyInstanceUID = series1.studyInstanceUID

        anon_study = study.anonymize()
        with tempfile.TemporaryDirectory() as d:
            anon_study.write(d, keep_uid=True, formats='dicom')
            study1 = Study(d)
            self.assertEqual(len(study), len(study1))
            for uid in range(len(study)):
                self.assertEqual(study[uid].shape, study1[uid].shape)


class TestPatient(unittest.TestCase):

    def test_read_patient(self):
        patient = Patient('data/dicom/cohort.zip?cohort/P2')

        for uid in patient:
            study = patient[uid]
            studyDescription = study.studyDescription
            self.assertIsNotNone(studyDescription)
            self.assertIsNotNone(study.studyDate)
            self.assertIsNotNone(study.studyID)
        self.assertIsNotNone(patient.patientID)

    def test_write_patient(self):
        patient = Patient('data/dicom/cohort.zip?cohort/P2')
        with tempfile.TemporaryDirectory() as d:
            patient.write(d)

    def test_anonymize_patient(self):
        patient = Patient('data/dicom/cohort.zip?cohort/P2')
        anon_patient = patient.anonymize()
        with tempfile.TemporaryDirectory() as d:
            anon_patient.write(d)
            patient1 = Patient(d)
            self.assertEqual(len(patient), len(patient1))
            self.assertEqual('ANONYMOUS', patient1.patientName)


class TestCohort(unittest.TestCase):

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

    def test_anonymize_cohort(self):
        cohort = Cohort('data/dicom/cohort.zip?cohort')
        anon_cohort = cohort.anonymize()
        with tempfile.TemporaryDirectory() as d:
            anon_cohort.write(d)
            cohort1 = Cohort(d)


if __name__ == '__main__':
    unittest.main()
