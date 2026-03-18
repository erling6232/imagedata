.. _Anonymization:

Anonymization
===============

DICOM data can be anonymized on the Series, Study, Patient or Cohort level.
Each object class provides an anonymize() method.

Common parameters include:

+---------------+------------------------------------------------------------+
|uid_table      |Translation table for UIDs                                  |
+---------------+------------------------------------------------------------+
|kwargs         |dict or parameters giving mapping rules for specific        |
|               |DICOM attributes                                            |
+---------------+------------------------------------------------------------+
