import unittest
import os.path
import tempfile
import numpy as np
import argparse
from pydicom.dataset import FileMetaDataset
from pynetdicom import (
    AE, evt,
    StoragePresentationContexts,
    PYNETDICOM_IMPLEMENTATION_UID,
    PYNETDICOM_IMPLEMENTATION_VERSION
)

# from .context import imagedata
import src.imagedata.cmdline as cmdline
import src.imagedata.formats as formats
import src.imagedata.transports as transports
from src.imagedata.series import Series

scpdir = None


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
        self.setupDicomSCP(port=11112)

    def setupDicomSCP(self, port=11112):
        global scpdir
        ae = AE()
        ae.ae_title = 'Temp'
        ae.supported_contexts = StoragePresentationContexts
        # ae.on_c_store = obj.on_c_store
        handlers = [(evt.EVT_C_STORE, self.handle_store)]
        # Returns a ThreadedAssociationServer instance
        self.server = ae.start_server(('localhost', port), block=False, evt_handlers=handlers)
        scpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        global scpdir
        self.server.shutdown()
        # noinspection PyUnresolvedReferences
        scpdir.cleanup()

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
            os.path.join(scpdir.name, ds.SOPInstanceUID),
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
            scpdir.name
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

    @unittest.skip("skipping test_transport_walk")
    def test_transport_walk(self):
        patID = '123456'
        stuInsUID = '1.2.3.4'
        serInsUID = '1.2.3.4.5'
        accno = '98765'
        transport = transports.Transport(
            'dicom://localhost:11112/Temp')
        for root, dirs, files in transport.walk('{}/*cerebrum*'.format(patID)):
            print(dirs, files)
            for dir in dirs:
                info = transport.info('{}/{}'.format(root, dir))
                print(info)
            break
        for root, dirs, files in transport.walk('{}/{}/*'.format(patID, stuInsUID)):
            print(dirs, files)
            for dir in dirs:
                info = transport.info('{}/{}'.format(root, dir))
                print(info)
            break
        for root, dirs, files in transport.walk('{}/{}/{}/*'.format(patID, stuInsUID, serInsUID)):
            print(dirs, files)
            for dir in dirs:
                print(dir)
            for filename in files:
                info = transport.info('{}/{}'.format(root, filename))
                print(filename)
            break
        transport.close()

    @unittest.skip("skipping test_transport_cget_series")
    def test_transport_cget_series(self):
        patID = '123456'
        stuInsUID = '1.2.3.4'
        serInsUID = '1.2.3.4.5'
        accno = '98765'
        serNum = 4
        si1 = Series('dicom://localhost:11112/Temp/{}/{}/{}'.format(
            patID, stuInsUID, serInsUID
        ))
        print(si1.shape, si1.spacing, si1.patientName, si1.accessionNumber, si1.seriesNumber)
        si2 = Series('dicom://localhost:11112/Temp/{}/{}/{}'.format(
            patID, accno, serNum
        ))
        print(si2.shape, si2.spacing, si2.patientName, si2.accessionNumber, si2.seriesNumber)
