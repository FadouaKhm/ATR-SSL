import cv2, json
import numpy as np
from config import config

def crop_patch(imPr, c_x, c_y, max_d):
    cr_im = imPr[max(c_y - max_d, 0):min(c_y + max_d, imPr.shape[0]),
                max(0, c_x - max_d):min(c_x + max_d, imPr.shape[1])]

    cropped_img = np.stack((cr_im, cr_im, cr_im), axis=-1)
    cropped_img = cv2.resize(cropped_img, (config['data_config']["input_size"], config['data_config']["input_size"]))

    return cropped_img

def area(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return (dx * dy) / ((b.xmax - b.xmin) * (b.ymax - b.ymin))
    else:
        return 0

def DecodeDet(dets):
    dets_ = [[det["confidence"], det["shape"]["data"]] for det in dets if det["confidence"]>0.01]
    return dets_

class ParamParser(object):
    def __init__(self, param_path):
        super(ParamParser, self).__init__()
        with open(param_path, 'r') as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as of:
            json.dump(self.__dict__, of, indent=4)

    @property
    def dict(self):
        return self.__dict__

import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)