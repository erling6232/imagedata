
import numpy as np
import logging

def compare_headers(self, hdr, newhdr):
    #logging.debug('compare_headers: name {} {}'.format(hdr.name, newhdr.name))
    self.assertEqual(hdr.name, newhdr.name)
    #logging.debug('compare_headers: description {} {}'.format(hdr.description, newhdr.description))
    self.assertEqual(hdr.description, newhdr.description)
    self.assertEqual(hdr.authors, newhdr.authors)
    self.assertEqual(hdr.version, newhdr.version)
    self.assertEqual(hdr.url, newhdr.url)
    self.assertEqual(hdr.input_order, newhdr.input_order)
    #self.assertEqual(hdr.sort_on, newhdr.sort_on)
    np.testing.assert_array_almost_equal(hdr.spacing, newhdr.spacing, decimal=4)
    logging.debug('compare_headers:    hdr.orientation={}'.format(hdr.orientation))
    logging.debug('compare_headers: newhdr.orientation={}'.format(newhdr.orientation))
    np.testing.assert_array_almost_equal(hdr.orientation, newhdr.orientation,
            decimal=4)

    self.assertEqual(hdr.imagePositions.keys(), newhdr.imagePositions.keys())
    for k in hdr.imagePositions.keys():
        logging.debug('compare_headers:    hdr.imagePositions[{}]={}'.format(k,hdr.imagePositions[k]))
        logging.debug('compare_headers: newhdr.imagePositions[{}]={}'.format(k,newhdr.imagePositions[k]))
        np.testing.assert_array_almost_equal(
                hdr.imagePositions[k],
                newhdr.imagePositions[k],
                decimal=4)

    try:
        np.testing.assert_array_equal(hdr.sliceLocations, newhdr.sliceLocations)
    except ValueError:
        pass
    # DicomHeaderDict[slice].tuple(tagvalue, filename, dicomheader)
    try:
        self.assertEqual(hdr.DicomHeaderDict.keys(), newhdr.DicomHeaderDict.keys())
        #for k in hdr.DicomHeaderDict.keys():
        #    self.assertEqual(hdr.DicomHeaderDict[k], newhdr.DicomHeaderDict[k])
    except ValueError:
        pass
    self.assertEqual(hdr.tags.keys(), newhdr.tags.keys())
    for k in hdr.tags.keys():
        np.testing.assert_array_equal(hdr.tags[k], newhdr.tags[k])
    try:
        self.assertEqual(hdr.seriesNumber, newhdr.seriesNumber)
        self.assertEqual(hdr.seriesDescription, newhdr.seriesDescription)
        self.assertEqual(hdr.imageType, newhdr.imageType)
    except ValueError:
        pass

    np.testing.assert_array_almost_equal(
            hdr.transformationMatrix,
            newhdr.transformationMatrix,
            decimal=4)

