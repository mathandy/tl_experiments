"""Copies a directory, then converts all PNG images to JPEG images.

Usage
-----
Typical usage::

    $ python convert_PNG_dataset_to_JPEGs.py MY_DATA_DIR MY_NEW_PNG_DATA_DIR


For more help::

    $ python convert_PNG_dataset_to_JPEGs.py -h


"""


import os
import cv2 as cv  # pip install opencv-python
from shutil import copytree


def is_JPEG(filename, extensions=('jpg', 'jpeg')):
    return os.path.splitext(filename)[-1][1:].lower() in extensions


def is_PNG(filename, extensions=('png',)):
    return os.path.splitext(filename)[-1][1:].lower() in extensions


def convert_PNG_dataset_to_JPEGs(input_dir, output_dir):
    copytree(input_dir, output_dir)
    for directory, _, filenames in os.walk(output_dir):
        for filename in filenames:
            if is_PNG(filename):
                dst = os.path.join(directory,
                                   os.path.splitext(filename)[0] + '.jpg')
                all_good = cv.imwrite(dst, cv.imread(os.path.join(directory, filename)))
                if all_good:
                    os.remove(os.path.join(directory, filename))
                else:
                    raise Exception("Trouble converting %s." % filename)
            else:
                continue


if __name__ == '__main__':
    # parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Copies a directory, then converts all PNG images to '
                    'JPEG images.')
    parser.add_argument("input_dir",
                        help="root directory to search for images.  Any "
                             "images in this directory, including inside "
                             "subdirectories, will be converted to a JPEGs "
                             "(and stored in output_dir).  This files in this "
                             "directory will not be changed unless "
                             "`output_dir == input_dir`.")
    parser.add_argument('output_dir',
                        help="Where to store the images.  Note the "
                             "directory structure of input_dir will be "
                             "preserved.")
    args = parser.parse_args()
    convert_PNG_dataset_to_JPEGs(args.input_dir, args.output_dir)
