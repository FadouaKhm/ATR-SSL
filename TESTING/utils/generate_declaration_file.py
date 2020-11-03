import os
from utils.utils import *
from utils.read_json_files import *
from datetime import datetime
from utils import arf
from config import config

def generate_declaration_file(test_files, output, model_path,  data_type, NT_class):
    today = datetime.now()
    if not os.path.exists(os.path.join("./results",model_path.split("/")[-1][:-4])):
        os.mkdir(os.path.join("./results", model_path.split("/")[-1][:-4]))
    output_path = os.path.join(os.path.join("./results", model_path.split("/")[-1][:-4]), today.strftime("%H_%M_%S_%d_%m_%Y"))
    os.mkdir(output_path)

    classes = np.unique(list(config["ClassMapping"].values()))
    class_id = dict(zip(classes, range(len(classes))))
    if NT_class == "True":
        class_id["NT"] = len(classes)
    num_classes = len(class_id)
    l=-1
    for i in range(len(test_files["path"])):
        o = arf.arf_open(os.path.join(config['data_config']['root'], test_files["path"][i] + ".arf"))
        im = o.load(0)
        decl_file = dict()
        decl_file['source'] = data_type
        decl_file['fileUID'] = test_files["path"][i].split("/")[-1]
        decl_file['frameDeclarations'] = dict()
        InputLines = GenInputLines([test_files["path"][i]], config['data_config']['root'],im.shape[0], im.shape[1])
        if data_type == "YOLO":
            decl_file_path = os.path.join(config['data_config']['root'], test_files["path"][i].replace('/arf/', '/json_decl/')) + '.decl.json'
            with open(decl_file_path, 'r') as fid:
                det_decl_file = json.load(fid)
            for j in range(1, len(InputLines), test_files["sampling"][i]):
                fr_nb = "f{}".format(j)
                if fr_nb not in det_decl_file["frameDeclarations"]:
                    pass
                else:
                    dets_ = DecodeDet(det_decl_file["frameDeclarations"][fr_nb]["declarations"])
                    decl_file['frameDeclarations']['f' + str(j)] = {'declarations':[]}
                    for bb in dets_:
                        l += 1

                        for tgt in range(num_classes):
                            if l >= len(output["pred_conf"]):
                                break
                            decl_file['frameDeclarations']['f'+str(j)]['declarations'].append(
                                {'confidence': 1.*np.float32(list(output["pred_conf"][l])[tgt]),
                                 'class': int(list(class_id.keys())[tgt]),
                                 'shape': {'data': bb[1], 'type': "bbox_xywh"}
                             })

        elif data_type == "GT":
            
            for j in range(1, len(InputLines), test_files["sampling"][i]):
                coords_bb = InputLines[j][2]
                decl_file['frameDeclarations']['f'+ str(j)] = {'declarations': []}
                for k in range(len(coords_bb)):
                    l += 1

                    bb_gt = coords_bb[k]
                    up_l_x = bb_gt[0]
                    up_l_y = bb_gt[1]
                    bot_r_x = bb_gt[2]
                    bot_r_y = bb_gt[3]
                    w = bot_r_x - up_l_x
                    h = bot_r_y - up_l_y
                    for tgt in range(num_classes):
                        if l >= len(output["pred_conf"]):
                            break
                        decl_file['frameDeclarations']['f'+str(j)]['declarations'].append(
                            {'confidence': 1.*np.float32(list(output["pred_conf"][l])[tgt]),
                             'class': int(list(class_id.keys())[tgt]),
                             'shape': {'data': [up_l_x, up_l_y, w, h], 'type': "bbox_xywh"}
                             })
        with open(os.path.join(output_path, test_files["path"][i].split("/")[-1]+".decl.json"), "w") as f:
            json.dump(decl_file, f)

