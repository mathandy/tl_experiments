"""Walks through a directory (recursively) making all images grayscale."""
import cv2 as cv
import os
from andnn_util import Timer
from shutil import copytree


def is_JPEG_or_PNG(fn, extensions=('jpg', 'jpeg', 'png')):
    return os.path.splitext(fn)[1][1:].lower() in extensions


def main(image_dir, out_dir):
    copytree(image_dir, out_dir)
    for folder, _, fns in os.walk(out_dir):
        for fn in fns:
            fn_full = os.path.join(folder, fn)
            img = cv.imread(fn_full, 0)  # read as grayscale
            try:
                img.shape
            except AttributeError:
                os.remove(fn_full)
                print("Removed %s as it couldn't be read." % fn_full)
                continue
            if is_JPEG_or_PNG(fn):
                out_name = fn_full
            else:
                out_name = fn_full + '.jpg'
            cv.imwrite(out_name, img)


if __name__ == '__main__':
    from sys import argv
    with Timer("Converting to grayscale"):
        main(*argv[1:])
