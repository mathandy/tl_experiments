"""Converts a directory of .npy files into a directory of images.

One subdirectory is created for each .npy file."""
from __future__ import division, print_function
import numpy as np
from time import time
import os


def make_npys(npy_dir, out_dir, samples_per_class):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    npys = [fn for fn in os.listdir(npy_dir) if fn.endswith('.npy')]
    for npy in npys:
        start_time = time()
        print("Working on %s ..." % npy)
        images = np.load(os.path.join(npy_dir, npy))
        np.random.shuffle(images)
        np.save(os.path.join(out_dir, npy), images[:samples_per_class])
        print("done (in %s s)." % (time() - start_time))


if __name__ == '__main__':
    
    NPY_DIR = '/home/andy/datasets/quickdraw/NPYs'
    OUT_DIR = '/home/andy/datasets/quickdraw/NPYs1k'
    SAMPLES_PER_CLASS = 100
    
    make_npys(NPY_DIR, OUT_DIR, SAMPLES_PER_CLASS)
