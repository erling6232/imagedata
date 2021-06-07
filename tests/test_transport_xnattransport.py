import unittest
from imagedata.series import Series
from imagedata.transports import Transport


class TestXNATTransport(unittest.TestCase):

    # @unittest.skip("skipping test_transport_walk")
    def test_transport_walk(self):
        # project = 'project'
        # subject = 'subject'
        # transport = Transport('xnat://xnat.server/{}/{}'.format(project, subject))
        # for root, dirs, files in transport.walk(''):
        #     for dir in dirs:
        #         info = transport.info('{}/{}'.format(root, dir))
        #         print(info)
        #     for filename in files:
        #         info = transport.info('{}/{}'.format(root, filename))
        #         print(info)
        pass

    # @unittest.skip("skipping test_get_series")
    def test_get_series(self):
        # si1 = Series('xnat://xnat.server/project/subject/experiment/scan')
        # si1.seriesNumber = 9999
        # si1.seriesDescription = 'tttt'
        # si1.write('xnat://xnat.server/project/subject/experiment')
        pass


if __name__ == '__main__':
    unittest.main()
