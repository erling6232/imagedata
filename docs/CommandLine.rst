.. _CommandLine:

*****************************
Console Applications
*****************************

Included console applications
===============================

dicom_dump
--------------

Scan folders of DICOM files, and report the structure.
Can be handy when the sorting of the images is unclear.

Example of output on a MR diffusion-weighted acquisition:

.. code-block::

    % dicom_dump ./EP2D_DIFF_B50_400_800_TRA_P2_TRACEW_DFC_MIX_0007
    StuInsUID 1.2.3.4.30000019020708423676500000016: 90 images
    SerInsUID 1.2.3.4.2019020714224132517098797.0.0.0: 90 images
    SerTim 142629.896000: 90 images
    SeqNam *ep_b50t: 30 images
    SeqNam *ep_b400t: 30 images
    SeqNam *ep_b800t: 30 images
    AcqNum 1: 90 images
    ImaTyp DERIVED: 90 images
    Echo 1: 90 images

dicom_sort
--------------

Sort DICOM files into folders for each patient, study and series.
'dicom_sort' will attempt to assign meaningful names to the folders.

Usage:
    dicom_sort <destination> <list of input directories and files>

image_calculator
---------------------
Calculate a new series based on existing series.

Usage:
    image_calculator <output> <expression> <inputs>

The inputs will be assigned variables a, b, c, *etc.*
The expression, like 'a+b', should result in a new series of similar
size as 'a'.
This series will be saved in the <output>.
The calculator can do simple math and np (NumPy) operations, as long
as the resulting image has a shape compatible with 'a'.

DICOM series typically result in uint16 data. To perform the calculation in
floating point, include option '\-\-dtype float64'.

Example calculating mean of three series. The input data are converted to float64:

.. code-block::

    % image_calculator --serdes 'Mean T1_VIBE_FLIP' --dtype float64 \
        mean '(a+b+c)/3' T1_VIBE_FLIP*
    Converting input...
    a = T1_VIBE_FLIP18_0013 (30, 192, 192) float64
    b = T1_VIBE_FLIP3_0011 (30, 192, 192) float64
    c = T1_VIBE_FLIP8_0012 (30, 192, 192) float64
    mean = (a+b+c)/3 (30, 192, 192) float64

Example creating a mask=1 of equal size to input data. Notice the
input data is only used to give matrix dimensions:

.. code-block::

    % image_calculator mask '1' T1_VIBE_FLIP8_0012/
    a = T1_VIBE_FLIP8_0012/ (30, 192, 192) uint16
    mask = 1 (30, 192, 192) uint16

Example creating a mask = np.eye(128). Notice there is no input data:

.. code-block::

    % image_calculator eye 'Series(np.eye(128))'
    eye = Series(np.eye(128)) (128, 128) float64

By default, np.eye() will produce float64 data as above. There are three methods to set the output to uint16:

.. code-block::

    % image_calculator eye 'Series(np.eye(128, dtype=np.uint16))'
    eye = Series(np.eye(128,dtype=np.uint16)) (128, 128) uint16

    % image_calculator --dtype uint16 eye 'Series(np.eye(128))'
    eye = Series(np.eye(128)) (128, 128) float64

In the latter case, the output float64 data is converted to uint16 when writing the output.

An existing DICOM object can be used as template to set DICOM attributes:

.. code-block::

    % image_calculator --template dicom/input --geometry dicom/input eye 'Series(np.eye(128))'
    eye = Series(np.eye(128)) (128, 128) float64

image_data
-----------------

Convert input to output data, possible modifying the image format. The input data can be a cohort of series,
study and patients. The output will be sorted in folders with appropriate names.

Usage:
    image_data [<options>] <output> <input list...>

Some options:
    \-\-of <output format>: Possible output format: dicom, itk, nifti, mat (default: dicom). Can be repeated.

    \-\-order <input order>: How to sort input files: time, b, fa, te (default: 'none').

    \-\-dtype <dtype>: Specify output datatype. Otherwise keep input datatype. Dtypes: uint8, uint16, int16, int, float, float32, float64, double.

    \-\-template <template>: Source to get DICOM template from.

    \-\-geometry <template>: Second DICOM template for geometry attributes.

    \-\-sernum <number>: Set DICOM series number.

    \-\-serdes <string>: Set DICOM series description.

    \-\-imagetype: Set DICOM imagetype, comma-separated. *E.g.* 'DERIVED,SECONDARY,MASK'

See "imagedata \-\-help" for other options.

Example:
First a DICOM volume is converted to ITK MetaImage.
Next, the ITK MetaImage is read, adding DICOM attributes from the original
DICOM volume.
After setting series number and description, the images are sent to PACS
using the DICOM protocol.

.. code-block:: bash

   # Convert dicom/volume to itk/Image.mha
   #                   Output        Input
   image_data --of itk itk/Image.mha dicom/volume

   # Convert itk/Image.mha to dicom using original data as template
   # Send to DICOM store
   image_data --of dicom \
       --template dicom/volume \
       --sernum 1000 \
       --serdes 'Series description' \
       --imagetype 'DERIVED,SECONDARY,MASK' \
       dicom://server:104/AETITLE # Output destination \
       itk/Image.mha              # Input data

image_list
-----------------

List available data on URL.

Example, recursive list on xnat-server:

.. code-block::

    % image_list -r xnat://xnat.local/Project/Subject/Experiment

image_show
-----------------

Display an image stack interactively.

Some options:
    \-\-order <input order>: How to sort input files (time, b, fa, te) (default: 'none').

The following controls can be used to manipulate the viewer:

* Mouse scroll wheel: scroll through the slices of a volume
* Array up/down: scroll through the slices of a volume
* Array left/right: step through the 4th dimension of a 4D dataset
* PgUp/PgDown: Page through datasets when there are more datasets than views
* Left mouse key pressed: adjust window/level of display
* Mouse over: will display 2D coordinate and signal intensity at mouse position
* 'q': Quit. Will end the console application.

image_statistics
---------------------

Describe series shape, dtype, min, max and mean.
The input data can be a cohort of series, study and patients.
The output will report on series parameters for each patient, study and series.

timeline
-------------

Print timeline for a dynamic acquisition

Write your own console applications
===================================

A command line program can be as simple as copying input to output,
selecting input and output formats by command line options.
See Figure for an example:

.. code-block:: python

   import argparse
   import imagedata.cmdline
   from imagedata import Series

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
transport images.
