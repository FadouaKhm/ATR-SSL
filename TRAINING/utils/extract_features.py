import cv2, pickle, json
import numpy as np
from utils import arf, preprocess
from utils.read_json_files import *
from collections import namedtuple
from random import seed, random
from scipy.ndimage.filters import gaussian_filter
from IPython import get_ipython
import matplotlib.pyplot as plt
from pylab import rcParams
from random import randint
# rcParams['figure.figsize'] = 10, 5 #sets figure size
# get_ipython().run_line_magic('matplotlib', 'inline')
from utils.prep_unlabeled import *

def crop_patch(imPr, c_x, c_y, up_l_x, up_l_y, max_d, shifted_data, expanded_data):
    if shifted_data:
        ratio_x = config["data_config"]["ratio_x"]
        ratio_y = config["data_config"]["ratio_y"]

        wx = 2 * np.abs(c_x - up_l_x)
        wy = 2 * np.abs(c_y - up_l_y)
        dx = np.int(ratio_x * wx)
        dy = np.int(ratio_y * wy)
        cr_im = imPr[max(c_y - max_d+dy, 0):min(c_y + max_d+dy, imPr.shape[0]),
                max(0, c_x - max_d+dx):min(c_x + max_d+dx, imPr.shape[1])]
    elif expanded_data:
        exp_ratio = config["data_config"]["expanded_BB"]

        wx = 2 * np.abs(c_x - up_l_x)
        wy = 2 * np.abs(c_y - up_l_y)
        dx = np.int(exp_ratio * wx)
        dy = np.int(exp_ratio * wy)
        cr_im = imPr[max(c_y - max_d-dy, 0):min(c_y + max_d+dy, imPr.shape[0]),
                max(0, c_x - max_d-dx):min(c_x + max_d+dx, imPr.shape[1])]
    else:
        cr_im = imPr[max(c_y - max_d,0):min(c_y + max_d, imPr.shape[0]), max(0,c_x - max_d):min(c_x + max_d,imPr.shape[1])]

    #cr_im = (cr_im - np.mean(cr_im)) / np.std(cr_im)
    cropped_img = np.stack((cr_im, cr_im, cr_im), axis=-1)
    cropped_img = cv2.resize(cropped_img, (config["data_config"]["input_size"], config["data_config"]["input_size"]))

    return cropped_img



def DecodeDet(dets):
    dets_ = [[det["confidence"], det["shape"]["data"]] for det in dets]
    return dets_

def folder_files(ImgsPaths):
    folder_files = {}
    for i in range(len(ImgsPaths)):
        folder = ImgsPaths[i][0]
        if folder not in folder_files.keys():
            folder_files[folder] = []
        folder_files[folder].append(i)

    return folder_files


def get_numPixels(mi_width, mi_height):
    width = 2 * np.abs(mi_width)
    height = 2 * np.abs(mi_height)
    numPixels = width * height

    return numPixels


def get_contrastMeasure(frame, up_l_x, up_l_y, c_x, c_y):
    wx = 2 * np.abs(up_l_x - c_x)
    wy = 2 * np.abs(up_l_y - c_y)

    max_d = np.maximum((c_y - up_l_y), (c_x - up_l_x))
    BiggerBB = frame[max(c_y - max_d,0):min(c_y + max_d, frame.shape[0]), max(0,c_x - max_d):min(c_x + max_d,frame.shape[1])]
    BiggerBB = (BiggerBB-np.min(BiggerBB))/(np.max(BiggerBB)-np.min(BiggerBB)) # normalize bigger BB

    BB = BiggerBB[BiggerBB.shape[0]//2-(c_y-up_l_y):BiggerBB.shape[0]//2+(c_y-up_l_y), BiggerBB.shape[1]//2-(c_x-up_l_x):BiggerBB.shape[1]//2+(c_x-up_l_x)]
    m1 = BB.mean()
    s1 = BB.std()
    paddedBB = np.pad(BB, [((BiggerBB.shape[0] - BB.shape[0]) // 2, (BiggerBB.shape[0] - BB.shape[0]) // 2),
                           ((BiggerBB.shape[1] - BB.shape[1]) // 2, (BiggerBB.shape[1] - BB.shape[1]) // 2)],
                      mode='constant')
    if BiggerBB.shape != paddedBB.shape:
        sh_x, sh_y = min(BiggerBB.shape[0], paddedBB.shape[0]) , min(BiggerBB.shape[1], paddedBB.shape[1])
        BiggerBB = BiggerBB[:sh_x, :sh_y]
        paddedBB = paddedBB[:sh_x, :sh_y]
        ring = BiggerBB - paddedBB
    else:
        ring = BiggerBB - paddedBB
    ring[BiggerBB.shape[0]//2-(c_y-up_l_y):BiggerBB.shape[0]//2+(c_y-up_l_y), BiggerBB.shape[1]//2-(c_x-up_l_x):BiggerBB.shape[1]//2+(c_x-up_l_x)] = np.nan

    ring_elements = ring[~np.isnan(ring)]
    
    if ring_elements.size == 0:
        contrast = 0
    else:

        m2 = ring_elements.mean()
        s2 = ring_elements.std()
    
        contrast = np.abs(m1 - m2) / (s1 + s2)
    return contrast


def area(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return (dx*dy)/((b.xmax-b.xmin)*(b.ymax-b.ymin))
    else:
        return 0

##### Get train and unlabeled #####
    
def get_range(filename, def_ranges): # = config["def_ranges"]
    return def_ranges['R'+filename.split('___R')[-1][0]]
##############    
def crop_blurred_BB_unlabeled(frame, up_l_x, up_l_y, c_x, c_y, blurr_ratio):
    import random    
    #n = random.uniform(0, 2)
    n = blurr_ratio
    #x = config['shared']['input_size']
    x = config["data_config"]["input_size"]
    wx = 2*np.abs(up_l_x - c_x)
    wy = 2*np.abs(up_l_y - c_y)
    w = max(wx,wy)

    #cr_im = frame[up_l_y:up_l_y-2*(up_l_y-c_y),up_l_x:up_l_x+2*(c_x-up_l_x)]
    cr_im = frame[up_l_y-(w-wy)//2:up_l_y-2*(up_l_y-c_y)+(w-wy)//2 , up_l_x-(w-wx)//2:up_l_x+2*(c_x-up_l_x)+(w-wx)//2]
    cr_im = gaussian_filter(cr_im, sigma = n)
#    cr_im = (cr_im-np.mean(cr_im))/np.std(cr_im)
    cropped_img = np.stack((cr_im,cr_im,cr_im),axis=-1)
    cropped_img = cv2.resize(cropped_img,(x,x))
    
    return cropped_img

def crop_expanded_BB_unlabeled(frame, up_l_x, up_l_y, c_x, c_y, scale_ratio):
    import random
    n = random.uniform(0, 1)-scale_ratio
    #x = config['shared']['input_size']
    x = config["data_config"]["input_size"]
    wx = np.abs(up_l_x - c_x)
    wy = np.abs(up_l_y - c_y)
    dx = np.int(2*n*wx)
    dy = np.int(2*n*wy)
    max_d = max(wx, wy)

    cr_im = frame[max(c_y - max_d-dy,0):min(c_y + max_d+dy, frame.shape[0]), max(0,c_x - max_d-dx):min(c_x + max_d+dx,frame.shape[1])]#frame[up_l_y-(w-wy)//2-dy:up_l_y-2*(up_l_y-c_y)+(w-wy)//2 +dy, up_l_x-(w-wx)//2-dx:up_l_x+2*(c_x-up_l_x)+(w-wx)//2+dx]


#    cr_im = (cr_im-np.mean(cr_im))/np.std(cr_im)
    cropped_img = np.stack((cr_im,cr_im,cr_im),axis=-1)
    cropped_img = cv2.resize(cropped_img,(x,x))
    
    return cropped_img


def crop_shifted_BB_unlabeled(frame, up_l_x, up_l_y, c_x, c_y, opts, shift_ratio):
    # if opt = "v" then vertical shifting, if opt = "h" then horizontal shifting, if opt = "both", the in both directions
    #seed(1)
    #bot_x = up_l_x - 2*(c_x-up_l_x)
    #bot_y = up_l_y - 2*(c_y-up_l_y)
    import random
    x = config["data_config"]["input_size"]
    n = random.uniform(0, 1) - shift_ratio
    wx = np.abs(up_l_x - c_x)
    wy = np.abs(up_l_y - c_y)
    max_d = max(wx,wy)
    dx = np.int(2*n*wx)+1
    dy = np.int(-1*2*n*wy)+1
    
    if opts == "v":
        cr_im = frame[max(c_y - max_d,0):min(c_y + max_d, frame.shape[0]), max(0,c_x - max_d-dx):min(c_x + max_d+dx,frame.shape[1])]
        #bb = [max(c_y - max_d,0), min(c_y + max_d, frame.shape[0]), max(0,c_x - max_d-dx), min(c_x + max_d+dx,frame.shape[1])]

        
    if opts == "h":
        cr_im = frame[max(c_y - max_d-dy,0):min(c_y + max_d+dy, frame.shape[0]), max(0,c_x - max_d):min(c_x + max_d,frame.shape[1])]


    if opts == "both":
        cr_im = frame[max(c_y - max_d-dy,0):min(c_y + max_d+dy, frame.shape[0]), max(0,c_x - max_d-dx):min(c_x + max_d+dx,frame.shape[1])]

        
#    cr_im = (cr_im-np.mean(cr_im))/np.std(cr_im)
    cropped_img = np.stack((cr_im,cr_im,cr_im),axis=-1)
    cropped_img = cv2.resize(cropped_img,(x,x))
    
    return cropped_img
############
def append_aug_tsne(unlabeled, v_img, imPr, up_l_x, up_l_y, c_x, c_y):
    unlabeled['Shifted_v']['images'].append(crop_shifted_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y,"v", config["gen_unlabeled"]['shift_v_ratio'] ))
    unlabeled['Shifted_h']['images'].append(crop_shifted_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y,"h", config["gen_unlabeled"]['shift_h_ratio'] ))
    unlabeled['Scaled']['images'].append(crop_expanded_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y, config["gen_unlabeled"]['scale_ratio'] ))
    unlabeled['Blurred']['images'].append(crop_blurred_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y, config["gen_unlabeled"]['blurr_filter']))                
    
         
    v_img.append(crop_shifted_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y,"v", config["gen_unlabeled"]['shift_v_ratio'] ))
    v_img.append(crop_shifted_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y,"h", config["gen_unlabeled"]['shift_h_ratio'] ))
    v_img.append(crop_expanded_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y, config["gen_unlabeled"]['scale_ratio'] ))
    v_img.append(crop_blurred_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y, config["gen_unlabeled"]['blurr_filter']))
    
    return unlabeled, v_img
def extract_udata_GT(data_files, training_sampling, OptRange, shifted_data=False, expanded_data=False):
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    root = config['data_config']['root']
    normalize_opts = config['normalize']

    # _IMG_DIR = './features/'+'/Target_IMG'
    # if not os.path.exists(_IMG_DIR ):
    #     os.mkdir(_IMG_DIR)
        
    unlabeled = dict()
    unlabeled['Shifted_v'] = dict()
    unlabeled['Shifted_v']['images'] = []

    unlabeled['Shifted_h'] = dict()
    unlabeled['Shifted_h']['images'] = []

    unlabeled['Scaled'] = dict()
    unlabeled['Scaled']['images'] = []

    unlabeled['Blurred'] = dict()
    unlabeled['Blurred']['images'] = []
    
    processed_imgs = []
    labels = []
    orig_labels = []
    aspects = []
    day_night = []
    contrasts = []
    bb_size = []
    hw_ratio = []
    mapped_px = []
    frame_id = []
    aug_id = []
    aug_type = []
    orig_class_id = []
    orig_labels_t = []
    v_img = []
    v_labels = []
    rnge = []
    for i in range(len(data_files)):            

        # Load Yolo file
        decl_file_path = os.path.join(root, data_files[i].replace('/arf/','/json_decl/'))+'.decl.json'
        with open(decl_file_path, 'r') as fid:
            decl_file = json.load(fid)
            
        o = arf.arf_open(os.path.join(root, data_files[i]+".arf"))
        im = o.load(0)
        InputLines = GenInputLines([data_files[i]], OptRange, im.shape[0]-1, im.shape[1]-1)
        if not InputLines:
            continue
        IMG_COUNT = 0
        for j in range(0, len(InputLines), training_sampling[i]):
            coords_bb = InputLines[j][2]
            im = o.load(InputLines[j][1])
            imPr, b_px1, w_px1, b_px2, w_px2 = preprocess.pre_process(im, config['invert'], config['clip1'], normalize_opts, config['clip2'])
            #print(config['clip1'])
            for k in range(len(coords_bb)):
                
                aspects = aspects + [InputLines[j][3][k]]*5

                if OptRange:
                    range_v = InputLines[j][4][0]
                else:
                    range_v = get_range(data_files[i], config["def_ranges"])
                rnge =  rnge +[range_v]*5
   
                day_night = day_night + [InputLines[j][5][0]]*5
                bb_gt = coords_bb[k]
                up_l_x   = bb_gt[0]
                up_l_y   = bb_gt[1]
                bot_r_x      = bb_gt[2]
                bot_r_y      = bb_gt[3]
                c_x = int(bot_r_x - (bot_r_x-up_l_x)//2)
                c_y = int(bot_r_y - (bot_r_y-up_l_y)//2)
                max_d = np.maximum((c_y - up_l_y), (c_x - up_l_x))
                cropped_img = crop_patch(imPr, c_x, c_y, up_l_x, up_l_y, max_d, shifted_data, expanded_data)
                # if j%2000 == 0 & IMG_COUNT<5:
                #     rcParams['figure.figsize'] = 10, 5 #sets figure size
                #     #get_ipython().run_line_magic('matplotlib', 'inline')
                #     plt.imshow(cropped_img[:,:,0], vmin =cropped_img[:,:,0].min(), vmax = cropped_img[:,:,0].max() )
                #     plt.savefig(os.path.join(_IMG_DIR, data_files[i].split('/')[-1]+ '_frame'+str(j)+".png"))
                #     IMG_COUNT = IMG_COUNT+1
                #plt.show()                

                #print(np.max(cropped_img), np.min(cropped_img))
                processed_imgs.append(cropped_img)

                labels.append(bb_gt[4])
                orig_labels = orig_labels + [(bb_gt[4])]*5

                contrasts = contrasts + [get_contrastMeasure(imPr, up_l_x, up_l_y, c_x, c_y)]*5
                                         #(imPr, c_x, c_y, max_d, config["data_config"]["expand_BB"])]*5
                bb_size = bb_size + [(bot_r_y-up_l_y)*(bot_r_x-up_l_x)]*5
                hw_ratio = hw_ratio + [(bot_r_y-up_l_y)/(bot_r_x-up_l_x)]*5
            
                
                v_img.append(cropped_img)
                v_labels = v_labels + [bb_gt[4]]*5                

                unlabeled, v_img = append_aug_tsne(unlabeled, v_img, imPr, up_l_x, up_l_y, c_x, c_y)
                frame_id = frame_id + [j]*5
                aug_id = aug_id +[-1]+[j]*4
                aug_type = aug_type + ['None', 'Shifted_v', 'Shifted_h', 'Scaled', 'Blurred'] 
                
                #rb = Rectangle(up_l_x, up_l_y, bot_r_x, bot_r_y) 
                processed_imgs, labels = get_yolo_FA(processed_imgs, labels, decl_file,j,o,coords_bb )
                # if bb_gt[4] != 'CLUTTER':
                #     generated_BBs, contrast_BBs = generate_3rd_class(im.shape, imPr, max_d, Rectangle, rb)
                #     for idx in range(len(generated_BBs)):
                #         bb = generated_BBs[idx]
                #         contrasts += [contrast_BBs[idx]]*5
                #         bb_size += [4*max_d*max_d]*5
                #         hw_ratio += [1]*5
                #         cropped_img = crop_patch_(imPr, bb[0], bb[1], max_d)
                #         processed_imgs.append(cropped_img)
                #         labels.append('CLUTTER+')
                #         v_img.append(cropped_img)
                #         v_labels = v_labels + ['CLUTTER+']*5    
                #         unlabeled, v_img = append_aug_tsne(unlabeled, v_img, imPr, up_l_x, up_l_y, c_x, c_y)
                #         frame_id += [j]*5
                #         aug_type = aug_type + ['None', 'Shifted_v', 'Shifted_h', 'Scaled', 'Blurred']
                #         aug_id = aug_id +[-1]+[j]*4
                #         orig_labels = orig_labels + ['CLUTTER+']*5
                #         day_night = day_night + [InputLines[j][5][0]]*5
                #         rnge =  rnge +[range_v]*5
                #         aspects = aspects + [InputLines[j][3][k]]*5
    
    pr_imgs  = np.array(processed_imgs).astype('float32')
    pr_imgs = np.transpose(pr_imgs, (0,3,1,2))

    lbls  = np.array(labels)
    lbls  = lbls.reshape((lbls.shape[0],1))
    lbls = lbls.squeeze().reshape((-1))

    data_set = {}
    data_set["images"] = pr_imgs
    data_set["labels"]  = lbls
    
    ul_labels = labels[:]
    
    ul_lbl_train  = np.array(ul_labels)
    ul_lbl_train  = ul_lbl_train.reshape((ul_lbl_train.shape[0],1))
    ul_lbl_train = ul_lbl_train.squeeze().reshape((-1))
    
    unlabeled['Shifted_v']['images'] = np.array(unlabeled['Shifted_v']['images']).astype('float32')
    unlabeled['Shifted_v']['images'] = np.transpose(unlabeled['Shifted_v']['images'],(0,3,1,2))
    unlabeled['Shifted_v']['labels'] = ul_lbl_train
    
    unlabeled['Shifted_h']['images'] = np.array(unlabeled['Shifted_h']['images']).astype('float32')
    unlabeled['Shifted_h']['images'] = np.transpose(unlabeled['Shifted_h']['images'],(0,3,1,2))
    unlabeled['Shifted_h']['labels'] = ul_lbl_train
    
    unlabeled['Scaled']['images'] = np.array(unlabeled['Scaled']['images']).astype('float32')
    unlabeled['Scaled']['images'] = np.transpose(unlabeled['Scaled']['images'],(0,3,1,2))
    unlabeled['Scaled']['labels'] = ul_lbl_train
    
    unlabeled['Blurred']['images'] = np.array(unlabeled['Blurred']['images']).astype('float32')
    unlabeled['Blurred']['images'] = np.transpose(unlabeled['Blurred']['images'],(0,3,1,2))
    unlabeled['Blurred']['labels'] = ul_lbl_train
    unlabeled = split_l_u(unlabeled['Shifted_v'], unlabeled['Shifted_h'], unlabeled['Scaled'], unlabeled['Blurred'], config["gen_unlabeled"]["Selector"], config["gen_unlabeled"]["percentage"])

    features = dict()
    features["GT_YOLO"] = [0]*len(frame_id)
    features["yolo_conf"] = [-1]*len(frame_id)
    features["Frame_ID"] = frame_id
    features["Class_ID"] = v_labels
    #features["Orig_labels"] = orig_labels
    features["Target Name"] = orig_labels
    features["Aug_IDX"] = aug_id
    features["Aug_Type"] = aug_type
    features["RangeRaw"] = rnge
    if OptRange:
        features["Range"] = [config["Ranges"][i] for i in np.digitize(rnge, np.array(config["Ranges"])+100)]
    else:
        features["Range"] = rnge
    features["Time"] = day_night
    features["Aspects"] = aspects
    features["Contrast"] = contrasts
    features["bb_size"] = bb_size
    features["hw_ratio"] = hw_ratio
    #features["mapped_px"] = mapped_px


    
    to_tsne = dict()
    to_tsne["images"] = np.transpose(np.array(v_img).astype('float32'), (0,3,1,2))
    to_tsne["labels"] = np.array(v_labels).reshape((np.array(v_labels).shape[0],1)).squeeze().reshape((-1))
    
    return data_set, unlabeled, features, to_tsne
    

#    _DATA_DIR = config["data_path"]
#    if not os.path.exists(_DATA_DIR):
#        os.mkdir(_DATA_DIR)
#
#    np.save(os.path.join(_DATA_DIR, "GT/{}".format(rge)), data_set)
#    np.save(os.path.join(_DATA_DIR, "GT/u{}".format(rge)), unlabeled)
#    with open(config["data_path"]+'/GT/features_{}.pickle'.format(rge), 'wb') as handle:
#        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

########################
def crop_patch_(imPr, c_x, c_y, max_d):
    cr_im = imPr[max(c_y - max_d,0):min(c_y + max_d, imPr.shape[0]), max(0,c_x - max_d):min(c_x + max_d,imPr.shape[1])]
    #cr_im = (cr_im - np.mean(cr_im)) / np.std(cr_im)
    cropped_img = np.stack((cr_im, cr_im, cr_im), axis=-1)
    cropped_img = cv2.resize(cropped_img, (32, 32))

    return cropped_img

def get_yolo_FA(processed_imgs, labels, decl_file,j,o,coords_gt ):


    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')   
    fr_nb = "f{}".format(j+1)
    if fr_nb not in decl_file["frameDeclarations"]:
        pass
    else:
        dets_ = DecodeDet(decl_file["frameDeclarations"][fr_nb]["declarations"])
        for bb in dets_:
            im = o.load(j)
            imPr, b_px1, w_px1, b_px2, w_px2 = preprocess.pre_process(im, config['invert'], config['clip1'], config['normalize'], config['clip2'])
            x = bb[1][0]
            y = bb[1][1]
            w = bb[1][2]
            h = bb[1][3]
            up_l_x = x
            up_l_y = y
            bot_r_x = x+w
            bot_r_y = y+h
            c_x = up_l_x+w//2
            c_y = up_l_y+h//2
            max_d = np.maximum(w, h)
            cropped_img = crop_patch(imPr, c_x, c_y, up_l_x, up_l_y, max_d, False, False)
            ra = Rectangle(up_l_x, up_l_y, bot_r_x, bot_r_y)
            overlap_bb = dict()
            for k in range(len(coords_gt)):
                bb_gt = coords_gt[k]
                up_l_x_gt = bb_gt[0]
                up_l_y_gt = bb_gt[1]
                bot_r_x_gt = bb_gt[2]
                bot_r_y_gt = bb_gt[3]
                rb = Rectangle(up_l_x_gt, up_l_y_gt, bot_r_x_gt, bot_r_y_gt)
                overlap_bb[area(ra, rb)] = [bb_gt[4], k]
                #print(bb_gt[4])
                #nb_gt[config["cls_id"][config["cls_id_inv"][bb_gt[4]]]] += 1
#                nb_gt[bb_gt[4]] += 1
            max_area = np.max(list(overlap_bb.keys()))
            if max_area < config['data_config']['yolo_thresh']:

                processed_imgs.append(cropped_img)
                #labels.append(config["cls_id"][config["cls_id_inv"][overlap_bb[max_area][0]]])
                labels.append('CLUTTER_y')
    return processed_imgs, labels

def extract_udata_yolo(data_files, training_sampling, OptRange , shifted_data=False, expanded_data=False):
    
    unlabeled = dict()
    unlabeled['Shifted_v'] = dict()
    unlabeled['Shifted_v']['images'] = []

    unlabeled['Shifted_h'] = dict()
    unlabeled['Shifted_h']['images'] = []

    unlabeled['Scaled'] = dict()
    unlabeled['Scaled']['images'] = []

    unlabeled['Blurred'] = dict()
    unlabeled['Blurred']['images'] = []
    rnge = []
    processed_imgs = []
    labels = []
    orig_labels = []
    aspects = []
    day_night = []
    contrasts = []
    bb_size = []
    hw_ratio = []
    mapped_px = []
    frame_id = []
    aug_id = []
    aug_type = []
    orig_class_id = []
    orig_labels_t = []
    v_img = []
    v_labels = []
    
    yolo_conf = []
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    root = config["data_config"]['root']
    normalize_opts = config['normalize']

    overlap_area = []
    #nb_gt = {0:0, 1:0}
    nb_gt = dict.fromkeys(list(config['ClassMapping'].keys())+['CLUTTER'] + ['CLUTTER_y'],0)
    nb_fa = 0
    nb_det = dict.fromkeys(list(config['ClassMapping'].keys())+['CLUTTER'] + ['CLUTTER_y'],0)
    for i in range(len(data_files)):

        o = arf.arf_open(os.path.join(root, data_files[i]) + '.arf')
        im = o.load(0)
        InputLines = GenInputLines([data_files[i]], OptRange, im.shape[0], im.shape[1])
        
        if not InputLines:
            continue
        # Load Yolo file
        decl_file_path = os.path.join(root, data_files[i].replace('/arf/','/json_decl/'))+'.decl.json'
        with open(decl_file_path, 'r') as fid:
            decl_file = json.load(fid)


        for j in range(0,len(InputLines),  training_sampling[i]):
            if OptRange:
                range_v = InputLines[j][4][0]
            else:
                range_v = get_range(data_files[i], config["def_ranges"])
            coords_gt = InputLines[j][2]
            fr_nb = "f{}".format(j+1)
            if fr_nb not in decl_file["frameDeclarations"]:
                pass
            else:
                dets_ = DecodeDet(decl_file["frameDeclarations"][fr_nb]["declarations"])
                for bb in dets_:
                    im = o.load(j)
                    imPr, b_px1, w_px1, b_px2, w_px2 = preprocess.pre_process(im, config['invert'], config['clip1'], normalize_opts, config['clip2'])
                    x = bb[1][0]
                    y = bb[1][1]
                    w = bb[1][2]
                    h = bb[1][3]
                    up_l_x = x
                    up_l_y = y
                    bot_r_x = x+w
                    bot_r_y = y+h
                    c_x = up_l_x+w//2
                    c_y = up_l_y+h//2
                    max_d = np.maximum(w, h)
                    cropped_img = crop_patch(imPr, c_x, c_y, up_l_x, up_l_y, max_d, shifted_data, expanded_data)
                    ra = Rectangle(up_l_x, up_l_y, bot_r_x, bot_r_y)
                    overlap_bb = dict()
                    for k in range(len(coords_gt)):
                        bb_gt = coords_gt[k]
                        up_l_x_gt = bb_gt[0]
                        up_l_y_gt = bb_gt[1]
                        bot_r_x_gt = bb_gt[2]
                        bot_r_y_gt = bb_gt[3]
                        rb = Rectangle(up_l_x_gt, up_l_y_gt, bot_r_x_gt, bot_r_y_gt)
                        overlap_bb[area(ra, rb)] = [bb_gt[4], k]
                        #print(bb_gt[4])
                        #nb_gt[config["cls_id"][config["cls_id_inv"][bb_gt[4]]]] += 1
                        nb_gt[bb_gt[4]] += 1
                    max_area = np.max(list(overlap_bb.keys()))
                    if max_area >= config['data_config']['yolo_thresh']:
#                        if data_files[i].split("/")[-1][5:9] in config["day_years"]:
#                            day_night = day_night +["day"]*5
#                        else:
#                            day_night = day_night +["night"]*5
                        yolo_conf = yolo_conf +[bb[0]]*5

                        rnge += [range_v]*5
                        #day_night = day_night + [get_day_night(data_files[i], config["def_day_night"])]*5
                        day_night = day_night + [InputLines[j][5][0]]*5
                        overlap_area = overlap_area + [max_area]*5
                        aspects = aspects + [InputLines[j][3][overlap_bb[max_area][1]]]*5
                        processed_imgs.append(cropped_img)
                        #labels.append(config["cls_id"][config["cls_id_inv"][overlap_bb[max_area][0]]])
                        labels.append(overlap_bb[max_area][0])
                        
                        v_img.append(cropped_img)
                        # v_labels = v_labels + [config["cls_id"][config["cls_id_inv"][overlap_bb[max_area][0]]]]*5
                        v_labels = v_labels + [overlap_bb[max_area][0]]*5
                        unlabeled, v_img = append_aug_tsne(unlabeled, v_img, imPr, up_l_x, up_l_y, c_x, c_y)

                        frame_id = frame_id + [j]*5
                        aug_id = aug_id +[-1]+[j]*4
                        aug_type = aug_type + ['None', 'Shifted_v', 'Shifted_h', 'Scaled', 'Blurred'] 

                        nb_det[labels[-1]] += 1
                        orig_labels_t.append(overlap_bb[max_area][0])
                        orig_labels = orig_labels + [overlap_bb[max_area][0]]*5
                        #orig_class_id = orig_class_id + [config["cls_id_inv"][overlap_bb[max_area][0]]]*5
                        contrasts = contrasts + [get_contrastMeasure(imPr, up_l_x, up_l_y, c_x, c_y)]*5
                        bb_size = bb_size + [(bot_r_y-up_l_y)*(bot_r_x-up_l_x)]*5
                        hw_ratio = hw_ratio + [(bot_r_y-up_l_y)/(bot_r_x-up_l_x)]*5

                        
                        if bb_gt[4] != 'CLUTTER':
                            generated_BBs, contrast_BBs = generate_3rd_class(im.shape, imPr, max_d, Rectangle, Rectangle(up_l_x, up_l_y, bot_r_x, bot_r_y)  )
                            for idx in range(len(generated_BBs)):
                                bb = generated_BBs[idx]
                                contrasts += [contrast_BBs[idx]]*5
                                bb_size += [4*max_d*max_d]*5
                                hw_ratio += [1]*5
                                cropped_img = crop_patch_(imPr, bb[0], bb[1], max_d)
                                processed_imgs.append(cropped_img)
                                labels.append('CLUTTER+')
                                v_img.append(cropped_img)
                                v_labels = v_labels + ['CLUTTER+']*5    
                                unlabeled, v_img = append_aug_tsne(unlabeled, v_img, imPr, up_l_x, up_l_y, c_x, c_y)
                                frame_id += [j]*5
                                aug_type = aug_type + ['None', 'Shifted_v', 'Shifted_h', 'Scaled', 'Blurred']
                                aug_id = aug_id +[-1]+[j]*4
                                orig_labels = orig_labels + ['CLUTTER+']*5
                                day_night = day_night + [InputLines[j][5][0]]*5
                                rnge =  rnge +[range_v]*5
                                aspects = aspects + [InputLines[j][3][k]]*5
                                yolo_conf = yolo_conf +[bb[0]]*5


                        #mapped_px.append([b_px1, w_px1, b_px2, w_px2])
                    else:
                        yolo_conf = yolo_conf +[bb[0]]*5

                        rnge += [range_v]*5
                        #day_night = day_night + [get_day_night(data_files[i], config["def_day_night"])]*5
                        day_night = day_night + [InputLines[j][5][0]]*5
                        overlap_area = overlap_area + [max_area]*5
                        aspects = aspects + [InputLines[j][3][overlap_bb[max_area][1]]]*5
                        processed_imgs.append(cropped_img)
                        #labels.append(config["cls_id"][config["cls_id_inv"][overlap_bb[max_area][0]]])
                        labels.append('CLUTTER_y')
                        
                        v_img.append(cropped_img)
                        # v_labels = v_labels + [config["cls_id"][config["cls_id_inv"][overlap_bb[max_area][0]]]]*5
                        v_labels = v_labels + ['CLUTTER_y']*5
                        unlabeled, v_img = append_aug_tsne(unlabeled, v_img, imPr, up_l_x, up_l_y, c_x, c_y)

                        frame_id = frame_id + [j]*5
                        aug_id = aug_id +[-1]+[j]*4
                        aug_type = aug_type + ['None', 'Shifted_v', 'Shifted_h', 'Scaled', 'Blurred'] 

                        nb_det[labels[-1]] += 1
                        orig_labels_t.append(overlap_bb[max_area][0])
                        orig_labels = orig_labels + [overlap_bb[max_area][0]]*5
                        #orig_class_id = orig_class_id + [config["cls_id_inv"][overlap_bb[max_area][0]]]*5
                        contrasts = contrasts + [get_contrastMeasure(imPr, up_l_x, up_l_y, c_x, c_y)]*5
                        bb_size = bb_size + [(bot_r_y-up_l_y)*(bot_r_x-up_l_x)]*5
                        hw_ratio = hw_ratio + [(bot_r_y-up_l_y)/(bot_r_x-up_l_x)]*5

    pr_imgs  = np.array(processed_imgs).astype('float32')
    pr_imgs = np.transpose(pr_imgs, (0,3,1,2))

    lbls  = np.array(labels)
    lbls  = lbls.reshape((lbls.shape[0],1))
    lbls = lbls.squeeze().reshape((-1))

    data_set = {}
    data_set["images"] = pr_imgs
    data_set["labels"]  = lbls

    ul_labels = labels[:]
    
    ul_lbl_train  = np.array(ul_labels)
    ul_lbl_train  = ul_lbl_train.reshape((ul_lbl_train.shape[0],1))
    ul_lbl_train = ul_lbl_train.squeeze().reshape((-1))
    
    unlabeled['Shifted_v']['images'] = np.array(unlabeled['Shifted_v']['images']).astype('float32')
    unlabeled['Shifted_v']['images'] = np.transpose(unlabeled['Shifted_v']['images'],(0,3,1,2))
    unlabeled['Shifted_v']['labels'] = ul_lbl_train
    
    unlabeled['Shifted_h']['images'] = np.array(unlabeled['Shifted_h']['images']).astype('float32')
    unlabeled['Shifted_h']['images'] = np.transpose(unlabeled['Shifted_h']['images'],(0,3,1,2))
    unlabeled['Shifted_h']['labels'] = ul_lbl_train
    
    unlabeled['Scaled']['images'] = np.array(unlabeled['Scaled']['images']).astype('float32')
    unlabeled['Scaled']['images'] = np.transpose(unlabeled['Scaled']['images'],(0,3,1,2))
    unlabeled['Scaled']['labels'] = ul_lbl_train
    
    unlabeled['Blurred']['images'] = np.array(unlabeled['Blurred']['images']).astype('float32')
    unlabeled['Blurred']['images'] = np.transpose(unlabeled['Blurred']['images'],(0,3,1,2))
    unlabeled['Blurred']['labels'] = ul_lbl_train

    unlabeled = split_l_u(unlabeled['Shifted_v'], unlabeled['Shifted_h'], unlabeled['Scaled'], unlabeled['Blurred'], config["gen_unlabeled"]["Selector"], config["gen_unlabeled"]["percentage"])

    features = dict()
#    features["nb_gt"] = nb_gt
#    features["nb_fa"] = nb_fa
#    features["nb_det"] = nb_det
    features["GT_YOLO"] = [1]*len(frame_id)
    features["yolo_conf"] = yolo_conf
    features["Frame_ID"] = frame_id
    features["Class_ID"] = v_labels
    #features["Orig_labels"] = orig_labels
    features["Target Name"] = orig_labels
    features["Aug_IDX"] = aug_id
    features["Aug_Type"] = aug_type
    features["RangeRaw"] = rnge
    features["Time"] = day_night
    features["Aspects"] = aspects
    features["Contrast"] = contrasts
    features["bb_size"] = bb_size
    features["hw_ratio"] = hw_ratio
    if OptRange:
        features["Range"] = [config["Ranges"][i] for i in np.digitize(rnge, np.array(config["Ranges"])+100)]
    else:
        features["Range"] = rnge
    #features["mapped_px"] = mapped_px




    to_tsne = dict()
    to_tsne["images"] = np.transpose(np.array(v_img).astype('float32'), (0,3,1,2))
    to_tsne["labels"] = np.array(v_labels).reshape((np.array(v_labels).shape[0],1)).squeeze().reshape((-1))
    
    return data_set, unlabeled, features, to_tsne

def extract_udata_both(data_files, training_sampling, OptRange, shifted_data=False, expanded_data=False):
    data_set_y, unlabeled_y, features_y, to_tsne_y = extract_udata_yolo(data_files, training_sampling, OptRange, shifted_data=False, expanded_data=False) 
    data_set_gt, unlabeled_gt, features_gt, to_tsne_gt = extract_udata_GT(data_files, training_sampling, OptRange, shifted_data=False, expanded_data=False) 

    features = {**features_y, **features_gt}
    for k in features_y.keys():
        features[k] = features_y[k] + features_gt[k]
    
    # unlabeled_y = split_l_u(unlabeled_y['Shifted_v'], unlabeled_y['Shifted_h'], unlabeled_y['Scaled'], unlabeled_y['Blurred'], [1,1,0,1], 0.5)
    # unlabeled_gt = split_l_u(unlabeled_gt['Shifted_v'], unlabeled_gt['Shifted_h'], unlabeled_gt['Scaled'], unlabeled_gt['Blurred'], [1,1,0,1], 0.5)

    unlabeled = dict()
    unlabeled['images'] = np.concatenate((unlabeled_y['images'],unlabeled_gt['images']),axis=0)
    unlabeled['labels'] = np.concatenate((unlabeled_y['labels'],unlabeled_gt['labels']),axis=0)

    data_set = dict()
    data_set['images'] = np.concatenate((data_set_y['images'],data_set_gt['images']),axis=0)
    data_set['labels'] = np.concatenate((data_set_y['labels'],data_set_gt['labels']),axis=0)
    
    to_tsne = dict()
    to_tsne['images'] = np.concatenate((to_tsne_y['images'],to_tsne_gt['images']),axis=0)
    to_tsne['labels'] = np.concatenate((to_tsne_y['labels'],to_tsne_gt['labels']),axis=0)
    
    return data_set, unlabeled, features, to_tsne

def split_train_val_test_10c(val_rng):
    val_set = np.load(config["data_path"]+"/GT/{}.npy".format(val_rng),allow_pickle=True).item()
    train_set = dict()
    for rng in range(1000, 4000, 500):
        if rng != val_rng:
            if train_set == {}:
                train_set = np.load(config["data_path"] + "/GT/{}.npy".format(rng), allow_pickle=True).item()
            else:
                new_set = np.load(config["data_path"] + "/GT/{}.npy".format(rng), allow_pickle=True).item()
                train_set["images"] = np.concatenate([train_set["images"], new_set["images"]], axis=0)
                train_set["labels"] = np.concatenate([train_set["labels"], new_set["labels"]], axis=0)
    rng = np.random.RandomState(1)
    indices = rng.permutation(len(train_set["images"]))
    train_set["images"] = train_set["images"][indices]
    train_set["labels"] = train_set["labels"][indices]

    with open(config["data_path"]+"/GT/features_{}.pickle".format(val_rng), "rb") as f:
        features = pickle.load(f)
    masks = [np.array(features["aspects"]) == 1, np.array(features["aspects"]) == 3, np.array(features["aspects"]) == 5, np.array(features["aspects"]) == 7]
    total_mask = masks[0] + masks[1] + masks[2] + masks[3]
    validation_set = {"images": val_set["images"][total_mask], "labels": val_set["labels"][total_mask]}
    test_set = {"images": val_set["images"][~total_mask], "labels": val_set["labels"][~total_mask]}

    return train_set, validation_set, test_set


def split_train_val_test_2c(val_rng):
    val_set = np.load(config["data_path"] + "/GT/{}.npy".format(val_rng), allow_pickle=True).item()
    train_set_sampled = dict()
    for rng in range(1000, 4000, 500):
        if rng != val_rng:
            with open(config["data_path"] + "/GT/features_{}.pickle".format(rng), "rb") as f:
                features = pickle.load(f)
            orig_labels = np.array(features["orig_labels"])
            if train_set_sampled == {}:
                train_set = np.load(config["data_path"] + "/GT/{}.npy".format(rng), allow_pickle=True).item()
                train_set_sampled = {"images": np.concatenate(
                    [train_set["images"][orig_labels == 0], train_set["images"][orig_labels == 1]], axis=0),
                                     "labels": np.concatenate(
                                         [train_set["labels"][orig_labels == 0], train_set["labels"][orig_labels == 1]],
                                         axis=0)}
                for i in range(2, 10):
                    train_set_sampled["images"] = np.concatenate(
                        [train_set_sampled["images"], train_set["images"][orig_labels == i][::4]], axis=0)
                    train_set_sampled["labels"] = np.concatenate(
                        [train_set_sampled["labels"], train_set["labels"][orig_labels == i][::4]], axis=0)
            else:
                train_set = np.load(config["data_path"] + "/GT/{}.npy".format(rng), allow_pickle=True).item()
                train_set_sampled["images"] = np.concatenate([train_set_sampled["images"], np.concatenate(
                    [train_set["images"][orig_labels == 0], train_set["images"][orig_labels == 1]], axis=0)], axis=0)
                train_set_sampled["labels"] = np.concatenate([train_set_sampled["labels"], np.concatenate(
                    [train_set["labels"][orig_labels == 0], train_set["labels"][orig_labels == 1]], axis=0)], axis=0)
                for i in range(2, 10):
                    train_set_sampled["images"] = np.concatenate(
                        [train_set_sampled["images"], train_set["images"][orig_labels == i][::4]], axis=0)
                    train_set_sampled["labels"] = np.concatenate(
                        [train_set_sampled["labels"], train_set["labels"][orig_labels == i][::4]], axis=0)

    rng = np.random.RandomState(1)
    indices = rng.permutation(len(train_set_sampled["images"]))
    train_set_sampled["images"] = train_set_sampled["images"][indices]
    train_set_sampled["labels"] = train_set_sampled["labels"][indices]

    with open(config["data_path"]+"/GT/features_{}.pickle".format(val_rng), "rb") as f:
        features = pickle.load(f)
    masks = [np.array(features["aspects"]) == 1, np.array(features["aspects"]) == 3, np.array(features["aspects"]) == 5, np.array(features["aspects"]) == 7]
    total_mask = masks[0] + masks[1] + masks[2] + masks[3]
    validation_set = {"images": val_set["images"][total_mask], "labels": val_set["labels"][total_mask]}
    test_set = {"images": val_set["images"][~total_mask], "labels": val_set["labels"][~total_mask]}

    return train_set_sampled, validation_set, test_set

def get_contrastMeasure_3(frame, cx, cy, max_d, exp_pct): 
    dxy = np.int(exp_pct*2*max_d)+1
    
    BB = frame[cy-max_d:cy+max_d,cx-max_d:cx+max_d]
    m1 = BB.mean()
    s1 = BB.std()
    
    paddedBB = np.pad(BB, [(dxy, dxy), (dxy, dxy)], mode='constant')
    BiggerBB = frame[cy-max_d-dxy:cy+max_d+dxy,cx-max_d-dxy:cx+max_d+dxy]
    ring = BiggerBB - paddedBB
    ring[dxy:ring.shape[0]-dxy,dxy:ring.shape[1]-dxy] = np.nan
    
    ring_elements = ring[~np.isnan(ring)]
    m2 = ring_elements.mean()
    s2 = ring_elements.std()
    
    contrast = np.abs(m1-m2)/(s1+s2)
    return contrast
def generate_3rd_class(img_shape, imPr, max_d, Rectangle, rb):
    i = 0
    contrast_BBs = []
    coords_BBs = []
    g = gencoordinates(img_shape[0], img_shape[1], max_d, 3)
    while i<20 :  
        
        (cx, cy) = next(g)
        ra = Rectangle(cx-max_d , cy-max_d , cx+max_d , cy+max_d)
        # if area(ra, rb) == 0:
        #     contrast_BBs.append(get_contrastMeasure(imPr, up_l_x, up_l_y, c_x, c_y))
        #     coords_BBs.append([cx,cy])
        #     i += 1
        try:
            if area(ra, rb) == 0:
                contrast_BBs.append(get_contrastMeasure_3(imPr, cx, cy, max_d, 0.1))
                coords_BBs.append([cx,cy])
                i += 1
        except:
            continue
        
    sorted_contrast = np.argsort(contrast_BBs)
    
    return np.array(coords_BBs)[sorted_contrast[-3:]], np.array(contrast_BBs)[sorted_contrast[-3:]]  

def gencoordinates(m, n, max_d, dd):
    seen = set()

    x, y = randint(max_d+dd, n-max_d-dd), randint(max_d+dd, m-max_d-dd)

    while True:
        seen.add((x, y))
        yield (x, y)
        x, y = randint(max_d, n-max_d), randint(max_d, m-max_d)
        while (x, y) in seen:
            x, y = randint(max_d, n-max_d), randint(max_d, m-max_d)