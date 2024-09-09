.. _Collections:

Collections
===========

Introduction
------------

In addition to the Series level, there are collection classes Study, Patient and Cohort
for the case when multiple Series will be handled.

Each of the collection classes will take a source, and sort the images into
appropriate instances. In each case, the input order of each Series will
be auto-detected. There is no way to set input order explicitly.

The collection classes can be indexed by UID key or by ordered integer key.
The order is in insertion order.

Note: At present, this works for DICOM data only.

Reading a Study of Multiple Series
----------------------------------

The Study class can be used to sort DICOM files according to SeriesInstanceUID.
The input order of each Series is auto-detected.

.. code-block:: python

    from imagedata import Study

    vibe, dce = None
    study = Study('data/dicom')
    for i, uid in enumerate(study):
        if study[i].seriesDescription == 'vibe':
            vibe = series
        ...
    If not (vibe and dce):
        raise ValueError('Some series not found in study.')

Study Attributes
~~~~~~~~~~~~~~~~

+-------------------------+-----------------------------+-------------------+
| Study property name     | DICOM Attribute Name        | Usage             |
+=========================+=============================+===================+
| studyDate               | StudyDate                   | datetime.datetime |
+-------------------------+-----------------------------+-------------------+
| studyTime               | StudyTime                   | datetime.datetime |
+-------------------------+-----------------------------+-------------------+
| studyDescription        | StudyDescription            | str               |
+-------------------------+-----------------------------+-------------------+
| studyID                 | StudyID                     | str               |
+-------------------------+-----------------------------+-------------------+
| studyInstanceUID        | StudyInstanceUID            | str               |
+-------------------------+-----------------------------+-------------------+
| referringPhysiciansName | ReferringPhysiciansName     | str               |
+-------------------------+-----------------------------+-------------------+
| generalEquipment        | Instance of                 |                   |
|                         | GeneralEquipment class      | GeneralEquipment  |
+-------------------------+-----------------------------+-------------------+

GeneralEquipment Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------+-----------------------------+-------+
| Study property name     | DICOM Attribute Name        | Usage |
+=========================+=============================+=======+
| manufacturer            | Manufacturer                | str   |
+-------------------------+-----------------------------+-------+
| manufacturersModelName  | ManufacturerModelName       | str   |
+-------------------------+-----------------------------+-------+
| stationName             | StationName                 | str   |
+-------------------------+-----------------------------+-------+
| deviceSerialNumber      | DeviceSerialNumber          | str   |
+-------------------------+-----------------------------+-------+
| softwareVersions        | SoftwareVersions            | str   |
+-------------------------+-----------------------------+-------+

Reading a Patient with Multiple Study Instances
-----------------------------------------------

.. code-block:: python

    from imagedata import Patient

    patient = Patient('data/dicom')
    for uid in patient:
        study = patient[uid]
        print(study.studyDate, study.studyTime)

Patient Attributes
~~~~~~~~~~~~~~~~~~

+-------------------------+-------------------------+----------------------+
| Patient property name   | DICOM Attribute Name    | Usage                |
+=========================+=========================+======================+
| patientName             | PatientName             | str                  |
+-------------------------+-------------------------+----------------------+
| patientID               | PatientID               | str                  |
+-------------------------+-------------------------+----------------------+
| patientBirthDate        | PatientBirthDate        | str                  |
+-------------------------+-------------------------+----------------------+
| patientSex              | PatientSex              | str                  |
+-------------------------+-------------------------+----------------------+
| patientAge              | PatientAge              | str                  |
+-------------------------+-------------------------+----------------------+
| patientSize             | PatientSize             | float                |
+-------------------------+-------------------------+----------------------+
| patientWeight           | PatientWeight           | float                |
+-------------------------+-------------------------+----------------------+
| qualityControlSubject   | QualityControlSubject   | str                  |
+-------------------------+-------------------------+----------------------+
| clinicalTrialSubject    | Instance of             |                      |
|                         | ClinicalTrialSubject    |                      |
|                         | class                   | ClinicalTrialSubject |
+-------------------------+-------------------------+----------------------+
| patientIdentityRemoved  | PatientIdentityRemoved  | str                  |
+-------------------------+-------------------------+----------------------+
| deidentificationMethod  | DeidentificationMethod  | str                  |
+-------------------------+-------------------------+----------------------+

ClinicalTrialSubject Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------------------------+---------------------------------------------------+-------+
| Patient property name                | DICOM Attribute Name                              | Usage |
+======================================+===================================================+=======+
|sponsorName                           |ClinicalTrialSponsorName                           |       |
+--------------------------------------+---------------------------------------------------+-------+
|protocolID                            |ClinicalTrialProtocolID                            |       |
+--------------------------------------+---------------------------------------------------+-------+
|protocolName                          |ClinicalTrialProtocolName                          |       |
+--------------------------------------+---------------------------------------------------+-------+
|siteID                                |ClinicalTrialSiteID                                |       |
+--------------------------------------+---------------------------------------------------+-------+
|siteName                              |ClinicalTrialSiteName                              |       |
+--------------------------------------+---------------------------------------------------+-------+
|subjectID                             |ClinicalTrialSubjectID                             |       |
+--------------------------------------+---------------------------------------------------+-------+
|subjectReadingID                      |ClinicalTrialSubjectReadingID                      |       |
+--------------------------------------+---------------------------------------------------+-------+
|protocolEthicsCommitteeName           |ClinicalTrialProtocolEthicsCommitteeName           |       |
+--------------------------------------+---------------------------------------------------+-------+
|protocolEthicsCommitteeApprovalNumber |ClinicalTrialProtocolEthicsCommitteeApprovalNumber |       |
+--------------------------------------+---------------------------------------------------+-------+

Reading a Cohort of Multiple Patient Instances
-----------------------------------------------

.. code-block:: python

    from imagedata import Cohort

    cohort = Cohort('data/dicom')
    for id in cohort:
        patient = cohort[id]
        print(patient.patientName, patient.patientID)

Cohort Attributes
~~~~~~~~~~~~~~~~~

At present no Cohort attributes are implemented.

+-------------------------+-------------------------+----------+
| Cohort property name    | DICOM Attribute Name    | Usage    |
+=========================+=========================+==========+
| N/A                     | N/A                     | Not used |
+-------------------------+-------------------------+----------+
