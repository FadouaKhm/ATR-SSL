import cv2, json
import numpy as np
from utils import arf, preprocess
from utils.read_json_files import *
from collections import namedtuple
from utils.utils import *
from config import config

def extract_data_yolo(data_files):
    classes = np.unique(list(config["ClassMapping"].values()))
    class_id = dict(zip(classes,range(len(classes))))
    class_id["NT"] = len(classes)

    processed_imgs = []
    labels = []
    for i in range(0, len(data_files["path"])):
        print(data_files["path"][i])
        o = arf.arf_open(os.path.join(config['data_config']['root'], data_files["path"][i] + ".arf"))
        im = o.load(0)
        InputLines = GenInputLines([data_files["path"][i]], config['data_config']['root'], im.shape[0], im.shape[1])
        # Load Yolo file
        decl_file_path = os.path.join(config['data_config']['root'], data_files["path"][i].replace('/arf/', '/json_decl/')) + '.decl.json'
        with open(decl_file_path, 'r') as fid:
            decl_file = json.load(fid)
        for j in range(1, len(InputLines), data_files["sampling"][i]):
            fr_nb = "f{}".format(j)
            if fr_nb not in decl_file["frameDeclarations"]:
                pass
            else:
                dets_ = DecodeDet(decl_file["frameDeclarations"][fr_nb]["declarations"])
                for bb in dets_:
                    im = o.load(j)
                    imPr = preprocess.pre_process(im, config['invert'], config['clip1'], config["normalize"],
                                                  config['clip2'])
                    x = bb[1][0]
                    y = bb[1][1]
                    w = bb[1][2]
                    h = bb[1][3]
                    up_l_x = x
                    up_l_y = y
                    c_x = up_l_x + w // 2
                    c_y = up_l_y + h // 2
                    max_d = np.maximum(w // 2, h // 2)
                    cropped_img = crop_patch(imPr, c_x, c_y, max_d)
                    processed_imgs.append(cropped_img)
                    labels.append(-1)

    return processed_imgs, labels

def extract_data_GT(data_files):
    classes = np.unique(list(config["ClassMapping"].values()))
    class_id = dict(zip(classes, range(len(classes))))
    class_id["NT"] = len(classes)

    processed_imgs = []
    labels = []
    for i in range(0, len(data_files["path"])):
        print(data_files["path"][i])
        o = arf.arf_open(os.path.join(config['data_config']['root'], data_files["path"][i] + ".arf"))
        im = o.load(0)
        InputLines = GenInputLines([data_files["path"][i]], config['data_config']['root'], im.shape[0], im.shape[1])
        for j in range(1, len(InputLines), data_files["sampling"][i]):
            coords_bb = InputLines[j][2]
            im = o.load(InputLines[j][1])
            imPr = preprocess.pre_process(im, config['invert'], config['clip1'], config["normalize"], config['clip2'])
            for k in range(len(coords_bb)):
                bb_gt = coords_bb[k]
                up_l_x = bb_gt[0]
                up_l_y = bb_gt[1]
                bot_r_x = bb_gt[2]
                bot_r_y = bb_gt[3]
                c_x = int(bot_r_x - (bot_r_x - up_l_x) // 2)
                c_y = int(bot_r_y - (bot_r_y - up_l_y) // 2)
                max_d = np.maximum((c_y - up_l_y), (c_x - up_l_x))
                cropped_img = crop_patch(imPr, c_x, c_y, max_d)
                processed_imgs.append(cropped_img)
                labels.append(class_id[config["ClassMapping"][bb_gt[4]]] if bb_gt[4]!="NT" else class_id[bb_gt[4]])
    return processed_imgs, labels

def get_test_data(data_type, data_files, NT_class):
    classes = np.unique(list(config["ClassMapping"].values()))
    class_id = dict(zip(classes,range(len(classes))))
    class_id["NT"] = len(classes)

    if data_type == "GT":
        print("EXTRACTING GT DATA ...")
        images, labels = extract_data_GT(data_files)

    elif data_type == "YOLO":
        print("EXTRACTING YOLO DATA ...")
        images, labels = extract_data_yolo(data_files)

    images = np.array(images).astype('float32')
    images = np.transpose(images, (0, 3, 1, 2))

    labels = np.array(labels)
    labels = labels.reshape((labels.shape[0], 1))
    labels = labels.astype(np.int32).squeeze().reshape((-1))

    if NT_class!="True":
        images = images[labels != class_id["NT"]]
        labels = labels[labels != class_id["NT"]]

    data_set = {}
    data_set["images"] = images
    data_set["labels"] = labels
    return data_set