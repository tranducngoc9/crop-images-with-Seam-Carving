
from ctypes.wintypes import RGB
import sys
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

def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in trange(c - new_c):
        img = carve_column(img)

    return img

def crop_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img


def insert_c(img, scale_c):
    temp = np.copy(img)
    
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in trange(new_c - c):
        img = insert_carve_column(temp, img)
        temp = carve_column(temp)
    return img

def insert_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = insert_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img


def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img


def insert_carve_column(temp, img):
    row, col, _ = img.shape

    M, backtrack = minimum_seam(temp)
    #mask = np.ones((row, col), dtype=np.bool)

    j = np.argmin(M[-1])
    output = np.zeros((row,col+1,3))
       
    for i in reversed(range(row)):

        for ch in range(3):
            if j == 0:
                p = np.average(img[i, j: j + 2, ch])
                output[i, j, ch] = img[i, j, ch]
                output[i, j + 1, ch] = p
                output[i, j + 1:, ch] = img[i, j:, ch]
            else:
                p = np.average(img[i, j - 1: j + 1, ch])
                output[i, : j, ch] = img[i, : j, ch]
                output[i, j, ch] = p
                output[i, j + 1:, ch] = img[i, j:, ch]
        
        j = backtrack[i, j]
    return output


def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

def main():
    sys.argv = ["hello.py" , "insert_c" , 1.5, "canh_dong.jpg" , "canh_dong_scale.jpg"]

    if len(sys.argv) != 5:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)
    
    which_axis = sys.argv[1]
    scale = float(sys.argv[2])
    in_filename = sys.argv[3]
    out_filename = sys.argv[4]

    img = imread(in_filename)

    if which_axis == 'crop_r':
        out = crop_r(img, scale)
    elif which_axis == 'crop_c':
        out = crop_c(img, scale)
    elif which_axis == 'insert_r':
        out = insert_r(img, scale)
    elif which_axis == 'insert_c':
        out = insert_c(img, scale)
    else:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)
    
    imwrite(out_filename, out)

if __name__ == '__main__':
    main()
