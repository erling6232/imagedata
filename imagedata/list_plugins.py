#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)

from imagedata.formats.abstractplugin import AbstractPlugin
import imagedata.formats as formats

if __name__ == '__main__':
    print("__main__")
    plugins = formats.get_plugins_list()
    print("Plugin Table 1")
    print("==============")
    for pname,ptype,pclass in plugins:
        print("%20s (%8s) %s" % (pname, ptype, pclass.description))

    formats.add_plugin_dir('/hus/home/eran/src/image_data.py/packaging/imagedata/imagedata/formats')
    plugins = formats.get_plugins_list()
    print("Plugin Table 2")
    print("==============")
    for pname,ptype,pclass in plugins:
        print("%20s (%8s) %s" % (pname, ptype, pclass.description))

    types = AbstractPlugin.__subclasses__()
    print("Plugin Table 3")
    print("==============")
    for plugin in types:
        print(plugin.__name__)
