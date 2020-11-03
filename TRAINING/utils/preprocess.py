import os
import numpy as np
from astropy.stats import median_absolute_deviation
import fnmatch
import itertools
from config import config


def getFilesNames(min_tr_rge, max_tr_rge, scenarios):
    flatten = itertools.chain.from_iterable
    files = os.listdir(os.path.join(r"D:\ATR Database", "cegr/json_truth"))
    files = ["cegr/arf/{}".format(f.split(".")[0]) for f in files]
    train_ranges = [rge for rge in [*scenarios] if rge <= max_tr_rge and rge >= min_tr_rge]
    FilesList = dict()
    for i in train_ranges:
        ptrn = scenarios[i]
        FilesList[i] = list(flatten([fnmatch.filter(files,"cegr/arf/"+p[:-4]) for p in ptrn]))

    return FilesList


def invert(frame):
    return ~frame


def clip1(frame, threshold):
    lims = np.percentile(frame, threshold)
    clipped_frame = np.maximum(frame, lims[0])
    b_px1 = len(clipped_frame[(clipped_frame == lims[0])]) * 100 / (clipped_frame.shape[0] * clipped_frame.shape[1])
    clipped_frame = np.minimum(lims[1], clipped_frame)
    w_px1 = len(clipped_frame[(clipped_frame == lims[1])]) * 100 / (clipped_frame.shape[0] * clipped_frame.shape[1])
    clipped_frame = (clipped_frame - clipped_frame.min()) / (clipped_frame.max() - clipped_frame.min()) * 255

    return clipped_frame, b_px1, w_px1


def clip2(frame, lims):
    clipped_frame = np.maximum(frame, lims[0])
    b_px2 = len(clipped_frame[(clipped_frame == lims[0])])*100/(clipped_frame.shape[0]*clipped_frame.shape[1])
    clipped_frame = np.minimum(lims[1], clipped_frame)
    w_px2 = len(clipped_frame[(clipped_frame == lims[1])]) * 100 / (clipped_frame.shape[0] * clipped_frame.shape[1])
    clipped_frame = (clipped_frame - clipped_frame.min()) / (clipped_frame.max() - clipped_frame.min()) * 255

    return clipped_frame, b_px2, w_px2


def normalize(frame, opts, params):
    if opts == 1:
        output = (frame - np.mean(frame)) / np.std(frame)

    if opts == 2:
        output = (frame - params['mean']) / params['std']

    if opts == 3:
        output = (frame - np.median(frame)) / median_absolute_deviation(frame)

    if opts == 4:
        output = (frame - params['median']) / params['mad']

    return output


def pre_process(frame, bool_invert, clip1_opts, normalize_opts, clip2_opts):
    b_px1 = -1
    w_px1 = -1
    b_px2 = -1
    w_px2 = -1
    if bool_invert == 1:
        frame = invert(frame)

    if clip1_opts['select'] == 1:
        frame, b_px1, w_px1 = clip1(frame, clip1_opts['threshold'])

    if normalize_opts['select'] == 1:
        frame = normalize(frame, normalize_opts['technique'], normalize_opts['averaged_frames'])

    if clip2_opts['select'] == 1:
        frame, b_px2, w_px2 = clip2(frame, clip2_opts['lims'])

    return frame, b_px1, w_px1, b_px2, w_px2








