#a.py này để test code
import sys
from traceback import print_tb
import cv2

from tqdm import trange
import numpy as np
from imageio.v2 import imread, imwrite
from scipy.ndimage import convolve
def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map

img = imread("anh_goc.jpg")
energy_map = calc_energy(img)
imwrite("energy_map.jpg", energy_map)
M = energy_map.copy()
print(energy_map.shape, M.shape)
backtrack = np.zeros_like(M, dtype=int)
print(backtrack)
print(backtrack.shape)