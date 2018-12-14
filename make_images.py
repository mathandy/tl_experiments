"""Converts a directory of .npy files into subdirectories of images.

One subdirectory is created for each .npy file."""


from __future__ import division, print_function
import numpy as np
import cv2 as cv
from time import time
import os


def make_images(npy_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for npy in os.listdir(npy_dir):
        subset = os.path.splitext(npy)[0]
        if os.path.exists(os.path.join(out_dir, subset)):
            continue
        os.mkdir(os.path.join(out_dir, subset))

        print("Working on %s ..." % npy, end='')
        time_npy_start = time()
        flat_images = np.load(os.path.join(npy_dir, npy))
        assert flat_images.shape[1] == 28**2
        images = flat_images.reshape(-1, 28, 28)

        for k, image in enumerate(images):
            cv.imwrite(os.path.join(out_dir, subset, subset + '_%s.jpg' % k),
                       image)
        print('Done (in %s s).' % (time() - time_npy_start))


if __name__ == '__main__':

    NPY_DIR = 'NPYs_first10'
    OUT_DIR = 'images_first10'

    make_images(NPY_DIR, OUT_DIR)
