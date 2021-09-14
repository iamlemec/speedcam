#!/usr/bin/env python3

import os
import toml
import argparse
import numpy as np
import cv2 as cv

from glob import glob
from pathlib import Path

def save_params(mtx, dist, path):
    data = {
        'K': mtx.tolist(),
        'D': dist.squeeze().tolist(),
    }
    with open(path, 'w+') as fid:
        toml.dump(data, fid)

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

# parse input arguments
parser = argparse.ArgumentParser(description='Calibrate camera lense distortion.')
parser.add_argument('images', type=str, help='glob pattern for image files')
parser.add_argument('output', type=str, default='params.toml', help='path to parameter output file')
parser.add_argument('--square_size', type=float, default=1.0, help='physical size of squares in cm')
parser.add_argument('--debug_dir', type=str, default=None, help='where to output test files')
args = parser.parse_args()

# get paths
img_names = glob(args.images)
debug_dir = Path(args.debug_dir) if args.debug_dir is not None else None

# ensure output directory
if debug_dir is not None and not os.path.isdir(debug_dir):
    os.mkdir(debug_dir)

# pattern characterization
pattern_size = 9, 6
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= args.square_size

# get default size
h0, w0 = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]

def processImage(fn):
    print(f'processing: {fn}')
    if (img := cv.imread(fn, 0)) is None:
        print('failed to load')
        return

    # enforce common size
    h, w = img.shape
    if (w, h) != (w0, h0):
        print(f'invalid size: {w} x {h}')

    # do the finding
    found, corners = cv.findChessboardCorners(img, pattern_size)
    if found:
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT), 30, 0.1
        cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

    # save marked up image
    if debug_dir is not None:
        vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawChessboardCorners(vis, pattern_size, corners, found)
        _path, name, _ext = splitfn(fn)
        fout = debug_dir / f'{name}_chess.png'
        cv.imwrite(str(fout), vis)

    if not found:
        print('chessboard not found')
        return

    return corners.reshape(-1, 2), pattern_points

# process all images
chessboards = [processImage(fn) for fn in img_names]
chessboards = [x for x in chessboards if x is not None]
img_points, obj_points = map(list, zip(*chessboards))

# calculate camera distortion
rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv.calibrateCamera(
    obj_points, img_points, (w0, h0), None, None
)

# display results
print()
print(f'RMS: {rms}')
print(f'camera matrix:\n{camera_matrix}')
print(f'distortion coefficients: {dist_coefs.ravel()}')

# save results
save_params(camera_matrix, dist_coefs, args.output)

# undistort the image with the calibration
for fn in img_names if debug_dir else []:
    _path, name, _ext = splitfn(fn)
    img_found = debug_dir / f'{name}_chess.png'
    outfile = debug_dir / f'{name}_undistorted.png'

    # load base image
    if (img := cv.imread(str(img_found))) is None:
        continue

    # undistort image
    h, w = img.shape[:2]
    newcam, roi = cv.getOptimalNewCameraMatrix(
        camera_matrix, dist_coefs, (w, h), 1, (w, h)
    )
    dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcam)

    # crop and save the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(str(outfile), dst)
