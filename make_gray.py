"""Walks through a directory (recursively) making all images grayscale."""
import cv2 as cv
import os
from andnn_util import Timer


# def is_image(fn, extensions=('jpg', 'jpeg', 'png')):
#     return os.path.splitext(fn)[1][1:].lower() in extensions


def main(image_dir, out_dir):
	for folder, _, fns in os.walk(image_dir):
		for fn in fns:
			fn_full = os.path.join(folder, fn)
			img = cv.imread(fn_full, 0)  # read as grayscale
			try:
				img.shape
			except AttributeError:
				continue
			cv.imwrite(fn_full.replace(image_dir, out_dir) + '.jpg', img)


if __name__ == '__main__':
	from sys import argv
	with Timer("Converting to grayscale"):
		main(*argv[1:])
