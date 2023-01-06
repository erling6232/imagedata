.. _Collections:

Collections
===========

Introduction
------------

In addition to the Series level, there are collection classes Cohort, Patient and Study
for the case when multiple Series will be handled.

Each of the collection classes will take a source, and sort the images into
appropriate instances. In each case, the input order of each Series will
be auto-detected. There is no way to set input order explicitly.

Note: At present, this works for DICOM data only.

Reading a study of multiple Series
----------------------------------

The Study class can be used to sort DICOM files according to SeriesInstanceUID.
The input order of each Series is auto-detected.

.. code-block:: python

    from imagedata import Study

    vibe, dce = None
    study = Study('data/dicom')
    for uid in study:
        series = study[uid]
        if series.seriesDescription == 'vibe':
            vibe = series
        ...
    If not (vibe and dce):
        raise ValueError('Some series not found in study.')

Study Attributes
----------------

Reading a patient with multiple Study instances
-----------------------------------------------

.. code-block:: python

    from imagedata import Patient

    patient = Patient('data/dicom')
    for uid in patient:
        study = patient[uid]
        print(study.studyDate, study.studyTime)

Reading a cohort of multiple Patient instances
-----------------------------------------------

.. code-block:: python

    from imagedata import Cohort

    cohort = Cohort('data/dicom')
    for id in cohort:
        patient = cohort[id]
        print(patient.patientName, patient.patientID)

