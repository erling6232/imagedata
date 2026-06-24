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


Example on Series level:

.. code-block:: python

    from imagedata import Series
    a = Series('data')

    # Explicit mapping rule(s)
    b = a.anonymize(patientName='ABCD')

    # dict of mapping rules
    kwargs = {
        'patientName': 'ABCD',
        'patientID': '126847'
    }
    c = a.anonymize(**kwargs)


The `uid_table` can be used to apply the same UIDs translations for each
anonymization. _E.g._:

.. code-block:: python

    from imagedata import Series
    a = Series('data1')
    b = Series('data2')

    uid_table = {}
    anon_a = a.anonymize(uid_table, patientName='ABCD')
    anon_b = b.anonymize(uid_table, patientName='ABCD')

