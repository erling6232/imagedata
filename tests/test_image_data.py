import unittest
import os.path
import tempfile
from collections import namedtuple
import pickle
import numpy as np
from numpy.random import default_rng
import copy
import pydicom.datadict

from src.imagedata.image_data import sort


class TestImageData(unittest.TestCase):

    def test_sort(self):
        with tempfile.TemporaryDirectory() as d:
            sort([d, 'data/dicom'])


if __name__ == '__main__':
    unittest.main()
