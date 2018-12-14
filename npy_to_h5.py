import numpy as np
import h5py
import os


npy_dir = 'NPYs'
h5_dir = 'H5s'


for npy in os.listdir(npy_dir):
    data = np.load(os.path.join(npy_dir, npy))
    h5 = h5py.File(os.path.join(h5_dir, os.path.splitext(npy)[0] + '.h5'))
    h5.create_dataset()
