import unittest
import logging

import src.imagedata as imagedata


class TestVersion(unittest.TestCase):
    # @unittest.skip("skipping test_version")
    def test_version(self):
        log = logging.getLogger("TestVersion.test_version")
        log.debug("test_version")

        logging.debug('test_version: version {}'.format(imagedata.__version__))
        if imagedata.__version__ == 'unknown':
            raise ValueError('Version number is {}'.format(imagedata.__version__))


if __name__ == '__main__':
    unittest.main()
