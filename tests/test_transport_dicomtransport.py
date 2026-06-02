import unittest
import os.path
import tempfile
import numpy as np
import argparse
from pydicom import dcmread
from pydicom.dataset import Dataset, FileMetaDataset
from pynetdicom import (
    AE, evt,
    debug_logger,
    QueryRetrievePresentationContexts,
    StoragePresentationContexts,
    PYNETDICOM_IMPLEMENTATION_UID,
    PYNETDICOM_IMPLEMENTATION_VERSION
)
from pynetdicom.sop_class import (
    StudyRootQueryRetrieveInformationModelGet,
    StudyRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelFind
)

import imagedata.cmdline as cmdline
import imagedata.formats as formats
import imagedata.transports as transports
from imagedata.series import Series
from imagedata.transports.dicomtransport import storage_presentation_contexts


scpdir = None


# debug_logger()


class TestDicomTransport(unittest.TestCase):


    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['dicom']

        self.opts_calling_aet = parser.parse_args(['--calling_aet', 'Temp'])
        if len(self.opts_calling_aet.output_format) < 1:
            self.opts_calling_aet.output_format = ['dicom']

        plugins = formats.get_plugins_list()
        self.dicom_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'dicom':
                self.dicom_plugin = pclass
        self.assertIsNotNone(self.dicom_plugin)
        self.setupDicomSCP()

    def setupDicomSCP(self, root=None, port=11112):
        global scpdir
        ae = AE()
        ae.ae_title = 'Temp'
        ae.supported_contexts = StoragePresentationContexts
        handlers = [(evt.EVT_C_STORE, self.handle_store)]
        # Returns a ThreadedAssociationServer instance
        self.server = ae.start_server(('localhost', port), block=False, evt_handlers=handlers)
        if root is None:
            scpdir = [tempfile.TemporaryDirectory()]
        else:
            scpdir = root

    def tearDown(self):
        global scpdir
        self.server.shutdown()
        for _dir in scpdir:
            if not issubclass(type(_dir), str):
                _dir.cleanup()

    # Implement a handler evt.EVT_C_STORE
    # @staticmethod
    # def on_c_store(ds, context, info):
    @staticmethod
    def handle_store(event):
        """Store the pydicom Dataset `ds`.

        Parameters
        ----------
        event : pydicom.event

        Returns
        -------
        status : int or pydicom.dataset.Dataset
            The status returned to the peer AE in the C-STORE response. Must be
            a valid C-STORE status value for the applicable Service Class as
            either an ``int`` or a ``Dataset`` object containing (at a
            minimum) a (0000,0900) *Status* element.
        """
        global scpdir
        ds = event.dataset
        context = event.context

        # logging.debug('on_c_store: info {}'.format(info))
        # calling_aet = info['requestor']['ae_title']
        # called_aet = info['acceptor']['ae_title']
        # logging.debug('on_c_store: calling_aet {}'.format(calling_aet))
        # logging.debug('on_c_store: called_aet {}'.format(called_aet))

        # Add the DICOM File Meta Information
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = ds.SOPClassUID
        meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        meta.ImplementationClassUID = PYNETDICOM_IMPLEMENTATION_UID
        meta.ImplementationVersionName = PYNETDICOM_IMPLEMENTATION_VERSION
        meta.TransferSyntaxUID = context.transfer_syntax

        # Add the file meta to the dataset
        ds.file_meta = meta

        # Save the dataset using the SOP Instance UID as the filename
        ds.save_as(
            os.path.join(scpdir[0].name, ds.SOPInstanceUID),
        )

        # Return a 'Success' status
        return 0x0000

    # @unittest.skip("skipping test_transport_single_image")
    def test_transport_single_image(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm')
        )
        self.assertEqual((192, 152), si1.shape)
        si1.write('dicom://localhost:11112/Temp', formats=['dicom'])
        si2 = Series(
            scpdir[0].name
        )
        # compare_headers(obj, si1, si2)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

    # @unittest.skip("skipping test_transport_single_image_calling_aet")
    def test_transport_single_image_calling_aet(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm')
        )
        si1.write(
            'dicom://localhost:11112/Temp',
            formats=['dicom'],
            opts=self.opts_calling_aet
        )


class TestDicomTransportWithDB(unittest.TestCase):

    instances = {}

    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['dicom']

        self.opts_calling_aet = parser.parse_args(['--calling_aet', 'Temp'])
        if len(self.opts_calling_aet.output_format) < 1:
            self.opts_calling_aet.output_format = ['dicom']

        plugins = formats.get_plugins_list()
        self.dicom_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'dicom':
                self.dicom_plugin = pclass
        self.assertIsNotNone(self.dicom_plugin)

    def setupDicomSCP(self, root=None, port=11112):
        global scpdir
        global instances

        ae = AE()
        ae.ae_title = 'Temp'
        ae.supported_contexts = StoragePresentationContexts  # QueryRetrievePresentationContexts
        # for context in storage_presentation_contexts:
        #     ae.add_supported_context(context)
        for context in ae.supported_contexts:
            context.scp_role = True
            context.scu_role = False
        ae.add_supported_context(StudyRootQueryRetrieveInformationModelGet)
        ae.add_supported_context(StudyRootQueryRetrieveInformationModelFind)
        ae.add_supported_context(PatientRootQueryRetrieveInformationModelFind)
        handlers = [(evt.EVT_C_GET, self.handle_get),
                    (evt.EVT_C_FIND, self.handle_find)]
        # Returns a ThreadedAssociationServer instance
        self.server = ae.start_server(('localhost', port), block=False, evt_handlers=handlers)
        if root is None:
            scpdir = [tempfile.TemporaryDirectory()]
        else:
            scpdir = root

        # Import stored SOP Instances
        instances = {}
        for fdir in scpdir:
            for fpath in os.listdir(fdir):
                instance = dcmread(os.path.join(fdir, fpath))
                instances[instance.SOPInstanceUID] = instance

    def tearDown(self):
        global scpdir
        self.server.shutdown()
        for _dir in scpdir:
            if not issubclass(type(_dir), str):
                _dir.cleanup()

    @staticmethod
    def handle_find(event):
        global scpdir
        global instances
        ds = event.identifier

        if 'QueryRetrieveLevel' not in ds:
            # Failure
            yield 0xC000, None
            return

        matching = instances.values()
        if 'PatientID' in ds and ds.PatientID not in ['*', '', '?', None]:
            _matching = [
                inst for inst in matching if inst.PatientID == ds.PatientID
            ]
            matching = _matching
        if 'StudyInstanceUID' in ds and ds.StudyInstanceUID not in ['*', '', '?', None]:
            _matching = [
                inst for inst in matching if inst.StudyInstanceUID == ds.StudyInstanceUID
            ]
            matching = _matching
        if 'SeriesInstanceUID' in ds and ds.SeriesInstanceUID not in ['*', '', '?', None]:
            _matching = [
                inst for inst in matching if inst.SeriesInstanceUID == ds.SeriesInstanceUID
            ]
            matching = _matching
        if 'SOPInstanceUID' in ds and ds.SOPInstanceUID not in ['*', '', '?', None]:
            _matching = [
                inst for inst in matching if inst.SOPInstanceUID == ds.SOPInstanceUID
            ]
            matching = _matching

        match ds.QueryRetrieveLevel:
            case 'PATIENT':
                _matching, _patients = [], []
                for inst in matching:
                    if inst.PatientID not in _patients:
                        _patients.append(inst.PatientID)
                        _matching.append(inst)
            case 'STUDY':
                _matching, _studies = [], []
                for inst in matching:
                    if inst.StudyInstanceUID not in _studies:
                        _studies.append(inst.StudyInstanceUID)
                        _matching.append(inst)
            case 'SERIES':
                _matching, _series = [], []
                for inst in matching:
                    if inst.SeriesInstanceUID not in _series:
                        _series.append(inst.SeriesInstanceUID)
                        _matching.append(inst)
            case 'IMAGE':
                _matching, _instances = [], []
                for inst in matching:
                    if inst.SOPInstanceUID not in _instances:
                        _instances.append(inst.SOPInstanceUID)
                        _matching.append(inst)
        matching = _matching

        for instance in matching:
            # Check if C-CANCEL has been received
            if event.is_cancelled:
                yield (0xFE00, None)
                return

            identifier = Dataset()
            # Copy requested elements
            for element in ds:
                if element.value in ['*', '', '?', None] and element.tag in instance:
                    identifier[element.tag] = instance[element.tag]
            identifier.QueryRetrieveLevel = ds.QueryRetrieveLevel

            # Pending
            yield (0xFF00, identifier)

    @staticmethod
    def handle_get(event):
        global scpdir
        global instances

        ds = event.identifier
        context = event.context
        meta = FileMetaDataset()

        if 'QueryRetrieveLevel' not in ds:
            # Failure
            yield 0xC000, None
            return

        if ds.QueryRetrieveLevel == 'SERIES':
            if 'SeriesInstanceUID' in ds:
                matching = [
                    inst for inst in instances.values() if inst.SeriesInstanceUID == ds.SeriesInstanceUID
                ]

                # Yield the total number of C-STORE sub-operations required
                yield len(matching)

                # Yield the matching instances
                for instance in matching:
                    # Check if C-CANCEL has been received
                    if event.is_cancelled:
                        yield (0xFE00, None)
                        return

                    # Pending
                    yield (0xFF00, instance)

    # @unittest.skip("skipping test_transport_walk")
    def test_transport_walk(self):
        si0 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'none',
            input_format='dicom')
        patID = si0.patientID
        stuInsUID = si0.studyInstanceUID
        serInsUID = si0.seriesInstanceUID
        si1 = Series(
            os.path.join('data', 'dicom', 'dwi'),
            'b',
            input_format='dicom')
        si2 = Series(
            os.path.join('data', 'dicom', 'cor_oblique'),
        'none',
            input_format='dicom')

        self.setupDicomSCP(root=[os.path.join('data', 'dicom', 'time', 'time00'),
                                 os.path.join('data', 'dicom', 'dwi'),
                                 os.path.join('data', 'dicom', 'cor_oblique')],
                           port=11112)

        transport = transports.Transport(
            'dicom://localhost:11112/Temp')
        # for root, dirs, files in transport.walk('{}/*cerebrum*'.format(patID)):
        expect = []
        expect.append([
            '/Temp/19.02.07-14:04:17-STD-1.3.12.2.1107.5.2.43.66035',
            '1.3.12.2.1107.5.2.43.66035.30000019020708423676500000016',
            0,
            '2019.02.07 140516.555000'
        ])
        expect.append([
            '/Temp/19.02.07-14:04:17-STD-1.3.12.2.1107.5.2.43.66035/1.3.12.2.1107.5.2.43.66035.30000019020708423676500000016',
            '2.16.578.1.37.1.1.2.8323329.32554.1549625089',
            '1.3.12.2.1107.5.2.43.66035.2019020714224132517098797.0.0.0',
            0,
            '#14: 0 MR fl3d_dynamic',
            '#7: 0 MR ep2d_diff_b50_400_800_tra_p2_TRACEW_DFC_MIX'
        ])
        expect.append([
            '/Temp/19.02.07-14:04:17-STD-1.3.12.2.1107.5.2.43.66035/1.3.12.2.1107.5.2.43.66035.30000019020708423676500000016/2.16.578.1.37.1.1.2.8323329.32554.1549625089',
            3
        ])
        expect.append([
            '/Temp/19.02.07-14:04:17-STD-1.3.12.2.1107.5.2.43.66035/1.3.12.2.1107.5.2.43.66035.30000019020708423676500000016/1.3.12.2.1107.5.2.43.66035.2019020714224132517098797.0.0.0',
            90
        ])
        i = 0
        for root, dirs, files in transport.walk('{}/*'.format(patID)):
            response = []
            response.append(root)
            for dir in dirs:
                response.append(dir)
            response.append(len(files))
            for dir in dirs:
                info = transport.info('{}/{}'.format(root, dir))
                response.append(info.strip())
            self.assertEqual(expect[i], response)
            i += 1

        expect = []
        expect.append([
            '/Temp/372372478',
            '1.3.12.2.1107.5.2.43.66035.30000025050805093675600000001',
            0,
            '2025.05.08 070936.739000  imagedata'
        ])
        expect.append([
            '/Temp/372372478/1.3.12.2.1107.5.2.43.66035.30000025050805093675600000001',
            '2.16.578.1.37.1.1.2.5056857de2.3033286.1747206092.145',
            0,
            '#7: 0 MR t2_tse_c_t_30'
        ])
        expect.append([
            '/Temp/372372478/1.3.12.2.1107.5.2.43.66035.30000025050805093675600000001/2.16.578.1.37.1.1.2.5056857de2.3033286.1747206092.145',
            40
        ])
        i = 0
        for root, dirs, files in transport.walk('{}/'.format(si2.patientID)):
            response = []
            response.append(root)
            for dir in dirs:
                response.append(dir)
            response.append(len(files))
            for dir in dirs:
                info = transport.info('{}/{}'.format(root, dir))
                response.append(info.strip())
            self.assertEqual(expect[i], response)
            i += 1

        expect = []
        expect.append([
            '/Temp/19.02.07-14:04:17-STD-1.3.12.2.1107.5.2.43.66035/1.3.12.2.1107.5.2.43.66035.30000019020708423676500000016',
            '2.16.578.1.37.1.1.2.8323329.32554.1549625089',
            '1.3.12.2.1107.5.2.43.66035.2019020714224132517098797.0.0.0',
            0,
            '#14: 0 MR fl3d_dynamic',
            '#7: 0 MR ep2d_diff_b50_400_800_tra_p2_TRACEW_DFC_MIX'
        ])
        expect.append([
            '/Temp/19.02.07-14:04:17-STD-1.3.12.2.1107.5.2.43.66035/1.3.12.2.1107.5.2.43.66035.30000019020708423676500000016/2.16.578.1.37.1.1.2.8323329.32554.1549625089',
            3
        ])
        expect.append([
            '/Temp/19.02.07-14:04:17-STD-1.3.12.2.1107.5.2.43.66035/1.3.12.2.1107.5.2.43.66035.30000019020708423676500000016/1.3.12.2.1107.5.2.43.66035.2019020714224132517098797.0.0.0',
            90
        ])
        i = 0
        for root, dirs, files in transport.walk('{}/{}/*'.format(patID, stuInsUID)):
            response = []
            response.append(root)
            for dir in dirs:
                response.append(dir)
            response.append(len(files))
            for dir in dirs:
                info = transport.info('{}/{}'.format(root, dir))
                response.append(info.strip())
            self.assertEqual(expect[i], response)
            i += 1

        expect = []
        expect.append([
            '/Temp/19.02.07-14:04:17-STD-1.3.12.2.1107.5.2.43.66035/1.3.12.2.1107.5.2.43.66035.30000019020708423676500000016/2.16.578.1.37.1.1.2.8323329.32554.1549625089',
            3,
            '',
            '',
            ''
        ])
        i = 0
        for root, dirs, files in transport.walk('{}/{}/{}/*'.format(patID, stuInsUID, serInsUID)):
            response = []
            response.append(root)
            for dir in dirs:
                response.append(dir)
            response.append(len(files))
            for dir in dirs:
                info = transport.info('{}/{}'.format(root, dir))
                response.append(info.strip())
            for filename in files:
                info = transport.info('{}/{}'.format(root, filename))
                response.append(info.strip())
            self.assertEqual(expect[i], response)
            i += 1
        transport.close()

    # @unittest.skip("skipping test_transport_cget_series")
    def test_transport_cget_series(self):
        si0 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'none',
            input_format='dicom')
        patID = si0.patientID
        stuInsUID = si0.studyInstanceUID
        serInsUID = si0.seriesInstanceUID
        accno = si0.accessionNumber
        serNum = si0.seriesNumber

        self.setupDicomSCP(root=[os.path.join('data', 'dicom', 'time', 'time00')],
                           port=11112)

        si1 = Series('dicom://localhost:11112/Temp/{}/{}/{}'.format(
            patID, stuInsUID, serInsUID
        ), input_format='dicom')
        self.assertEqual((3, 192, 152), si1.shape)
        np.testing.assert_array_almost_equal((3, 2.0833, 2.0833), si1.spacing, decimal=4)
        self.assertEqual(14, si1.seriesNumber)
        si2 = Series('dicom://localhost:11112/Temp/{}/{}/{}'.format(
            patID, accno, serNum
        ), input_format='dicom')
        self.assertEqual((3, 192, 152), si1.shape)
        np.testing.assert_array_almost_equal((3, 2.0833, 2.0833), si2.spacing, decimal=4)
