#!/usr/bin/env python3

import unittest
import sys
import os.path
import shutil
import numpy as np
import logging
import argparse
from pydicom.dataset import Dataset
from pynetdicom import (
    AE,
    StoragePresentationContexts,
    PYNETDICOM_IMPLEMENTATION_UID,
    PYNETDICOM_IMPLEMENTATION_VERSION
)

from .context import imagedata
import imagedata.cmdline
import imagedata.readdata
import imagedata.formats
from imagedata.series import Series
from .compare_headers import compare_headers


def on_c_store(ds, context, info):
    """Store the pydicom Dataset `ds`.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The dataset that the peer has requested be stored.
    context : namedtuple
        The presentation context that the dataset was sent under.
    info : dict
        Information about the association and storage request.

    Returns
    -------
    status : int or pydicom.dataset.Dataset
        The status returned to the peer AE in the C-STORE response. Must be
        a valid C-STORE status value for the applicable Service Class as
        either an ``int`` or a ``Dataset`` object containing (at a
        minimum) a (0000,0900) *Status* element.
    """
    logging.debug('on_c_store: info {}'.format(info))
    calling_aet = info['requestor']['ae_title']
    called_aet = info['acceptor']['ae_title']
    logging.debug('on_c_store: calling_aet {}'.format(calling_aet))
    logging.debug('on_c_store: called_aet {}'.format(called_aet))

    # Add the DICOM File Meta Information
    meta = Dataset()
    meta.MediaStorageSOPClassUID = ds.SOPClassUID
    meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    meta.ImplementationClassUID = PYNETDICOM_IMPLEMENTATION_UID
    meta.ImplementationVersionName = PYNETDICOM_IMPLEMENTATION_VERSION
    meta.TransferSyntaxUID = context.transfer_syntax

    # Add the file meta to the dataset
    ds.file_meta = meta

    # Set the transfer syntax attributes of the dataset
    ds.is_little_endian = context.transfer_syntax.is_little_endian
    ds.is_implicit_VR = context.transfer_syntax.is_implicit_VR

    # Save the dataset using the SOP Instance UID as the filename
    if not os.path.isdir('ttscp'):
        os.makedirs('ttscp')
    ds.save_as(
        os.path.join('ttscp', ds.SOPInstanceUID),
        write_like_original=False)

    # Return a 'Success' status
    return 0x0000


class TestDicomTransport(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        #sys.argv[1:] = ['aa', 'bb']
        #self.opts = parser.parse_args(['--order', 'none'])
        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['dicom']

        self.opts_calling_aet = parser.parse_args(['--calling_aet', 'Temp'])
        if len(self.opts_calling_aet.output_format) < 1:
            self.opts_calling_aet.output_format=['dicom']

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname,ptype,pclass in plugins:
            if ptype == 'dicom': self.dicom_plugin = pclass
        self.assertIsNotNone(pclass)

        self.setupDicomSCP(port=11112)

    def setupDicomSCP(self, port=11112):
        ae = AE()
        ae.ae_title = 'Temp'
        ae.supported_contexts = StoragePresentationContexts
        ae.on_c_store = on_c_store

        # Returns a ThreadedAssociationServer instance
        self.server = ae.start_server(('localhost', port), block=False)


    def tearDown(self):
        self.server.shutdown()
        shutil.rmtree('ttscp', ignore_errors=True)
        shutil.rmtree('ttd4d', ignore_errors=True)

    #@unittest.skip("skipping test_transport_single_image")
    def test_transport_single_image(self):
        si1 = Series(
            'data/dicom/time/time00/Image_00000.dcm'
            )
        si1.write('dicom://localhost:11112/Temp', formats=['dicom'])
        si2 = Series(
            'ttscp'
        )
        compare_headers(self, si1, si2)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

    #@unittest.skip("skipping test_transport_single_image_calling_aet")
    def test_transport_single_image_calling_aet(self):
        si1 = Series(
            'data/dicom/time/time00/Image_00000.dcm'
        )
        si1.write(
            'dicom://localhost:11112/Temp',
            formats=['dicom'],
            opts=self.opts_calling_aet
        )

if __name__ == '__main__':
    unittest.main()
