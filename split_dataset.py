""" Copies 5 images from each class to val set.

Important Notes
---------------
* Removes any files encountered that can't be read with `cv.imread()`.

Usage
-----
Copy entire dataset to data/train, then run this script.

E.g.::

    $ mkdir data-split
    $ mkdir data-split/train
    $ mkdir data-split/val
    $ cp -r data/* data-split/
    $ python split_dataset.py data-split


"""


import os
import cv2 as cv
import numpy as np


def split_dataset(data_dir, out_dir, n_val):
    n_val = int(n_val)
    os.mkdir(out_dir)
    train_dir = os.path.join(out_dir, 'train')
    val_dir = os.path.join(out_dir, 'val')
    os.mkdir(train_dir)
    os.mkdir(val_dir)
    os.system('cp -r %s/* %s/train' % (data_dir, out_dir))
    for subdir in os.listdir(train_dir):
        os.mkdir(os.path.join(val_dir, subdir))

    for subset in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, subset)):
            continue

        # remove unreadable images
        images = [f for f in os.listdir(os.path.join(train_dir, subset))]
        for fn in images:
            img = cv.imread(os.path.join(train_dir, subset, fn))
            try:
                img.shape
            except AttributeError:
                os.remove(os.path.join(train_dir, subset, fn))

        # move `n_val` images of this subset to
        images = [f for f in os.listdir(os.path.join(train_dir, subset))]
        for fn in np.random.choice(images, n_val, replace=False):
            os.rename(os.path.join(train_dir, subset, fn),
                      os.path.join(val_dir, subset, fn))


if __name__ == '__main__':
    from sys import argv
    split_dataset(*argv[1:])
