import unittest
import tempfile
import os.path
from collections import defaultdict
from imagedata.collection import Study
from imagedata.formats.dicomlib.instance import Instance


class TestDicomROI(unittest.TestCase):
    def process_content(self, result: defaultdict, content: Instance, finding: bool=False):  # -> list[dict]:
        try:
            result['value_type'] = content.ValueType
            for concept_name in content.ConceptNameCodeSequence:
                for attr in ['CodeValue', 'CodeMeaning']:  # , 'CodeSchemeDesignator']:
                    result[attr] = getattr(concept_name, attr)
            match result['CodeMeaning']:
                case 'Findings':
                    if not finding:
                        res = defaultdict(list)
                        self.process_content(res, content, finding=True)
                        result['Findings'].append(res)
                        return
                case 'Finding':
                    if not finding:
                        res = defaultdict(list)
                        self.process_content(res, content, finding=True)
                        result['Finding'].append(res)
                        return
            for attr in ['TextValue', 'ReferencedSOPSequence', 'GraphicData', 'GraphicType']:
                try:
                    result[attr] = getattr(content, attr)
                except AttributeError:
                    pass
            if 'MeasuredValueSequence' in content:
                result['num'] = []
                for num in content.MeasuredValueSequence:
                    result['num'].append(num.NumericValue)
                for unit in num.MeasurementUnitsCodeSequence:
                    result['num_unit'] = unit.CodeValue

            if 'ContentSequence' in content:
                for _cont in content.ContentSequence:
                    res = defaultdict(list)
                    self.process_content(res, _cont)
                    code_meaning = res['CodeMeaning']
                    if res['CodeMeaning'] in result:
                        print(f'Duplicate {res['CodeMeaning']}')

                    if res['CodeMeaning'][:7] == 'Finding':
                        result[res['CodeMeaning']].append(res[res['CodeMeaning']])
                    else:
                        result[res['CodeMeaning']].append(res)

            match content.ValueType:
                case 'TEXT':
                    pass
                case 'IMAGE':
                    pass
                case 'CONTAINER':
                    pass
                case 'NUM':
                    pass
                case 'SCOORD':
                    pass
                case _:
                    pass
        except AttributeError as e:
            raise
        return

    def test_roi(self):
        original = None
        evidences = None
        report = None
        result = None
        study = Study(os.path.join('data', 'dicom', 'cor_oblique_roi'),
                      input_format='dicom', skip_broken_series=True,
                      accept_duplicate_tag=True)
        for uid in study:
            print(uid, study[uid].seriesDescription)
            ser_desc = study[uid].seriesDescription
            sop_class = study[uid].SOPClassUID
            if ser_desc == 't2_tse_c_t_30':
                original = study[uid]
            elif sop_class == '1.2.840.10008.5.1.4.1.1.88.33':
                evidences = study[uid].header.datasets
            elif ser_desc[:13] == 'Result series':
                result = study[uid]
                # result = study[uid].header.datasets
            elif ser_desc == 'Report Data':
                report = study[uid].header.datasets
        if original is None:
            raise ValueError('No dynamic series found')
        if evidences is None:
            raise ValueError('No SR series found')

        pass
        results = []
        for evidence in evidences:
            result = defaultdict(list)
            self.process_content(result, evidence)
            results.append(result)
            pass


if __name__ == '__main__':
    unittest.main()
