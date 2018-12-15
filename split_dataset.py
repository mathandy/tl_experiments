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


def split_dataset(data_dir):
    subsets = [f for f in os.listdir(os.path.join(data_dir, 'train'))
               if os.path.isdir(f)]
    for subset in subsets:
        images = [f for f in os.listdir(os.path.join(data_dir, 'train', subset))]
        for fn in images:
            img = cv.imread(os.path.join(data_dir, 'train', subset, fn))
            try:
                img.shape
            except AttributeError:
                os.remove(os.path.join(data_dir, 'train', subset, fn))

        images = [f for f in os.listdir(os.path.join(data_dir, 'train', subset))]
        for fn in np.random.choice(images, 5, replace=False):
            os.rename(os.path.join(data_dir, 'train', subset, fn),
                      os.path.join(data_dir, 'val', subset, fn))


if __name__ == '__main__':
    from sys import argv
    assert len(argv) == 2
    split_dataset(argv[1])
