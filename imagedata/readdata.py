#!/usr/bin/env python3

"""Read/Write image files, calling appropriate transport, archive and format plugins
"""

# Copyright (c) 2013-2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import logging
import mimetypes
import argparse
import urllib.parse
import numpy as np
import imagedata.formats
import imagedata

class NoTransportError(Exception): pass
class NoArchiveError(Exception): pass

def read(urls, order=None, opts=None):
    """Read image data, calling appropriate transport, archive and format plugins

    Input:
    - urls: list of urls or url to read (list of str, or str)
    - order: determine how to sort the images (default: auto-detect)
    - opts: input options (argparse.Namespace or dict)
    Output:
    - hdr: header instance
    - si[tag,slice,rows,columns]: numpy array
    """

    logging.debug("reader.read: urls {}".format(urls))
    transport,my_urls,files = sanitize_urls(urls)
    if len(my_urls) < 1:
        raise ValueError("No URL(s) where given")
    logging.debug("reader.read: transport {} my_urls {}".format(transport,my_urls))

    # Let in_opts be a dict from opts
    if opts is None:
        in_opts = {}
    elif issubclass(type(opts),dict):
        in_opts = opts
    elif issubclass(type(opts), argparse.Namespace):
        in_opts = vars(opts)
    else:
        raise UnknownOptionType('Unknown opts type ({})'.format(type(opts)))

    # Let the calling party override a default input order
    input_order = imagedata.formats.INPUT_ORDER_NONE
    if 'input_order' in in_opts: input_order = in_opts['input_order']
    if order is not None: input_order = order
    logging.info("Input order: {}.".format(imagedata.formats.input_order_to_str(input_order)))

    # Pre-fill DICOM template
    pre_hdr = None
    if 'template' in in_opts and in_opts['template']:
        logging.debug("readdata.read template {}".format(in_opts['template']))
        template_transport,template_urls,template_files = sanitize_urls(in_opts['template'])
        reader = imagedata.formats.find_plugin('dicom')
        pre_hdr,_ = reader.read_headers(template_urls, template_files, input_order, in_opts)

        # Pre-fill DICOM geometry
        geom_hdr = None
        if 'geometry' in in_opts and in_opts['geometry']:
            logging.debug("readdata.read geometry {}".format(in_opts['geometry']))
            geometry_transport,geometry_urls,geometry_files = sanitize_urls(in_opts['geometry'])
            reader = imagedata.formats.find_plugin('dicom')
            geom_hdr,_ = reader.read_headers(geometry_urls, geometry_files, input_order, in_opts)
            add_dicom_geometry(pre_hdr, geom_hdr)

    # Call reader plugins in turn to read the image data
    plugins = sorted_plugins_dicom_first(imagedata.formats.get_plugins_list())
    logging.debug("readdata.read plugins length {}".format(len(plugins)))
    for pname,ptype,pclass in plugins:
        logging.debug("%20s (%8s) %s" % (pname, ptype, pclass.description))
        reader = pclass()
        try:
            hdr, si = reader.read(my_urls, files, pre_hdr, input_order, in_opts)
            #logging.debug("reader.read: hdr.imageType: {}".format(hdr['imageType']))

            return hdr, si
        except imagedata.formats.NotImageError as e:
            logging.warning("Giving up {}: {}".format(ptype,e))
            pass
        except Exception as e:
            logging.warning("Giving up (OTHER) {}: {}".format(ptype,e))
            pass

    if issubclass(type(urls),list):
        raise imagedata.formats.UnknownInputError("Could not determine input format of %s." % urls[0])
    else:
        raise imagedata.formats.UnknownInputError("Could not determine input format of %s." % urls)

def add_dicom_geometry(pre_hdr, geometry):
        """For each slice in geometry, use most of pre_hdr, adding a few attributes from geometry
        """

        #logging.debug("add_dicom_geometry template %s geometry %s" % (
        #    imagedata.formats.shape_to_str(self.shape),
        #    imagedata.formats.shape_to_str(geometry.shape)))
        pre_hdr['sliceLocations'] = geometry['sliceLocations'].copy()
        pre_hdr['spacing']        = geometry['spacing'].copy()
        pre_hdr['orientation']    = geometry['orientation'].copy()
        pre_hdr['imagePositions'] = {}
        logging.debug("add_dicom_geometry:")
        logging.debug("add_dicom_geometry: geometry.imagePositions {}".format(geometry['imagePositions'].keys()))
        for k in geometry['imagePositions'].keys():
            pre_hdr['imagePositions'][k] = geometry['imagePositions'][k].copy()

def write(si, dirname_template, filename_template, opts=None, formats=None):
    """Write image data, calling appropriate format plugins

    Input:
    - si[tag,slice,rows,columns]: Series array
    - dirname_template: output directory name
    - filename_template: template including %d for image number
    - opts: Output options (argparse.Namespace or dict)
    - formats: list of output formats, overriding opts.output_format (list or
      str)
    """

    #logging.debug("write: dir(si): {}".format(dir(si)))

    # Let out_opts be a dict from opts
    if opts is None:
        out_opts = {}
    elif issubclass(type(opts),dict):
        out_opts = out_opts
    elif issubclass(type(opts), argparse.Namespace):
        out_opts = vars(opts)
    else:
        raise UnknownOptionType('Unknown opts type ({})'.format(type(opts)))

    if 'sernum' in out_opts and out_opts['sernum']: si.seriesNumber = out_opts['sernum']
    if 'serdes' in out_opts and out_opts['serdes']: si.seriesDescription = out_opts['serdes']
    if 'imageType' in out_opts and out_opts['imageType']: si.imageType = out_opts['imageType']
    if 'frame' in out_opts and out_opts['frame']: si.frameOfReferenceUID = out_opts['frame']

    # Default output format is input format
    output_formats = [si.input_format]
    logging.debug("Default    output format : {}".format(output_formats))
    logging.debug("Overriding output formats: {}".format(formats))
    logging.debug("Options: {}".format(out_opts))
    if formats is not None:
        if isinstance(formats, list):
            output_formats = formats
        elif isinstance(formats, str):
            output_formats = [formats]
        else:
            raise TypeError("List of output format is not list() ({})".format(type(formats)))
    elif 'output_format' in out_opts and len(out_opts['output_format']):
        output_formats = out_opts['output_format']
    logging.info("Output formats: {}".format(output_formats))

    # Determine output dtype
    write_si = si
    if 'dtype' in out_opts and out_opts['dtype'] is not None:
        #if str_to_dtype(out_opts['dtype']) != si.dtype:
        if out_opts['dtype'] != si.dtype:
            #write_si = si.astype(str_to_dtype(out_opts['dtype']))
            write_si = si.astype(out_opts['dtype'])
    #logging.debug("write: dir(write_si): {}".format(dir(write_si)))

    # Call plugin writers in turn to store the data
    logging.debug("Available plugins {}".format(len(imagedata.formats.get_plugins_list())))
    written = False
    for pname,ptype,pclass in imagedata.formats.get_plugins_list():
        logging.debug("Attempt plugin {}".format(ptype))
        if ptype in output_formats:
            # Create plugin to write data in specified format
            writer = pclass()
            logging.debug("Created writer plugin of type {}".format(type(writer)))
            dirname = dirname_template.replace('%p', ptype)
            if write_si.ndim == 4 and write_si.shape[0] > 1:
                # 4D data
                writer.write_4d_numpy(write_si, dirname, filename_template, out_opts)
            elif write_si.ndim >= 3:
                # 3D data
                writer.write_3d_numpy(write_si, dirname, filename_template, out_opts)
            else:
                raise ValueError("Don't know how to write image of shape {}".format(write_si.shape))
            written = True
    if not written:
        raise ValueError("No writer plugin was found for {}".format(output_formats))

def sorted_plugins_dicom_first(plugins):
    """Sort plugins such that any Nifti plugin is used early."""
    for pname,ptype,pclass in plugins:
        if ptype == 'nifti':
            plugins.remove((pname,ptype,pclass))
            plugins.insert(0, (pname,ptype,pclass))
            break
    """Sort plugins such that any DICOM plugin is used first."""
    for pname,ptype,pclass in plugins:
        if ptype == 'dicom':
            plugins.remove((pname,ptype,pclass))
            plugins.insert(0, (pname,ptype,pclass))
            break
    return(plugins)

def sanitize_urls(urls):
    """Determine transport, archive and file from each url.
    
    Handle both single url, a url tuple, and a url list

    Input:
    - urls: list, tuple or single string, e.g.:
            file://dicom
              transport: file, url: dicom
            file://dicom.zip
              transport: file, url: zip://dicom.zip
            file://dicom.tar.gz
              transport: file, url: tar://dicom.zip
            http://server:port/dicom.zip
              transport: http, url: zip://dicom.zip
            dicomscp://server:port/AET
              transport: dicomscp, utl: AET
            xnat://server:port
             transport: xnat, utl: ?
    Output:
    - loc: for each url (dict)
      - transport: name of trasnport plugin
      - url
      - files
    """

    if isinstance(urls, list):
        my_urls = urls
    elif isinstance(urls, tuple):
        my_urls = list(urls)
    else:
        my_urls = [urls]

    # Map the url schemes
    loc = {}
    schemes = {}
    for url in my_urls:
        urldict = urllib.parse.urlsplit(url, scheme="file")
        schemes[urldict.scheme] = True
        loc[url] = {}
        loc[url]['transport'] = urldict.scheme
        loc[url]['url']       = urldict.path

    # Only file: scheme can be simplified
    #if len(schemes)>1 or "file" not in schemes: return (None,my_urls,None)
    if len(schemes)>1 or "file" not in schemes: return (None,loc,None)

    # Collect all paths and simplify
    paths = []
    for url in my_urls:
        urldict = urllib.parse.urlsplit(url, scheme="file")
        logging.debug("sanitize_urls: urldict {}".format(urldict))
        paths.append(urldict.path)
    cpath = os.path.commonpath(paths)
    if os.path.isfile(cpath):
        cpath = os.path.dirname(cpath)
    logging.debug("sanitize_urls: cpath {}".format(cpath))
    my_url = cpath
    logging.debug("sanitize_urls: url {}".format(url))

    # Calculate file names relative to cpath
    files = []
    for url in my_urls:
        urldict = urllib.parse.urlsplit(url, scheme="file")
        files.append( os.path.relpath(urldict.path, cpath))

    # Catch the case when no basename were given
    if files[0] == ".": files = None

    logging.debug("sanitize_urls: my_url {} {}".format(my_url,files))
    logging.debug("sanitize_urls: my_url {}".format(urllib.parse.urlsplit(my_url, scheme="file")))
    return ('file',[my_url],files)

def str_to_dtype(s):
    """Convert dtype string to numpy dtype."""
    return eval('np.'+s)
