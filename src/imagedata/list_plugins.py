#!/usr/bin/env python3

import os
import sys
import logging
from .formats.abstractplugin import AbstractPlugin
from .formats import get_plugins_list

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)
# logger.setLevel(logging.ERROR)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("sys.path = \n{}".format(sys.path))


if __name__ == '__main__':
    print("__main__")
    plugins = get_plugins_list()
    print("Plugin Table List")
    print("=================")
    for pname, ptype, pclass in plugins:
        print("%20s (%8s) %s" % (pname, ptype, pclass.description))

    # formats.add_plugin_dir('/hus/home/eran/src/image_data.py/packaging/imagedata/imagedata/formats')
    # plugins = formats.get_plugins_list()
    # print("Plugin Table 2")
    # print("==============")
    # for pname,ptype,pclass in plugins:
    #     print("%20s (%8s) %s" % (pname, ptype, pclass.description))

    types = AbstractPlugin.__subclasses__()
    print("Subclasses of AbstractPlugin")
    print("============================")
    for plugin in types:
        print(plugin.__name__)
