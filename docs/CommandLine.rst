.. _CommandLine:

Imagedata on the Command Line
=============================

A command line program can be as simple as copying input to output,
selecting input and output formats by command line options.
See Figure for an example:

.. code-block:: python

   import argparse
   import imagedata.cmdline
   from imagedata.series import Series

   if __name__ == '__main__':
       parser = argparse.ArgumentParser()
       imagedata.cmdline.add_argparse_options(parser)
       parser.add_argument("out_name", help="Output URL")
       parser.add_argument("in_dirs", nargs='+', help="Input URL")
       args = parser.parse_args()

       try:
           si = Series(args.in_dirs, opts=args)
       except Exception as e:
           print('Could not read {}: {}’.format(args.in_dirs, e))

       si.write(args.out_name, opts=args)

This python script can be called from the command line to convert and
transport images, like in Figure 5. First a DICOM volume is converted to
ITK MetaImage. Next, the ITK MetaImage is read, adding DICOM attributes
from the original DICOM volume. After setting series number and
description, the images are sent to PACS using the DICOM protocol.

.. code-block:: bash

   # Convert dicom/volume to itk/Image.mha
   image_data --of itk \
       itk/Image.mha # Output \
       dicom/volume  # Input


   # Convert itk/Image.mha to dicom using original data as template
   # Send to DICOM store
   image_data --of dicom \
       --template dicom/volume \
       --sernum 1000 \
       --serdes 'Series description' \
       --imagetype 'DERIVED,SECONDARY,MASK' \
       dicom://server:104/AETITLE # Output destination \
       itk/Image.mha              # Input data
