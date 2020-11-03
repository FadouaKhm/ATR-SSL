import numpy as np
import cv2, pickle, os
from random import randint
from sklearn.utils import shuffle
from collections import namedtuple
from config import config
from utils.YOLOFileIO import *
from utils import arf
from utils.preprocess import *
from pathlib import Path
import random
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageFont, ImageDraw
import cv2   
import copy 
from utils.test import test

##############################################################################
def DrawBoxes1(img,dets_,gt):
    
    draw=ImageDraw.Draw(img)
    det_colors=[(255,0,0),(255,255,0),(0,255,0),(0,255,255)]
    for bb in dets_:
        draw.rectangle(bb[2:],outline=det_colors[bb[1]])
    for bb in gt:
        draw.rectangle(bb[:-1],outline=(0,0,255))

def DrawBoxes(imPr, bboxes, filename, target_name, GTorYOLO, frameID, aug):
    
    
    im=imPr.astype(np.uint8)
    im=np.stack([im,im,im],axis =-1)
    img=Image.fromarray(im, 'RGB')
    draw=ImageDraw.Draw(img)
    for bb in bboxes:
        draw.rectangle(bb,outline=(255,0,0))
        
    img.save("./features/target_images/"+filename+'_'+target_name+'_' + GTorYOLO +'_f'+str(frameID)+'_' + aug +'.jpg','JPEG') 

            
            
def crop_patch_(imPr, c_x, c_y, max_d):
    cr_im = imPr[max(c_y - max_d,0):min(c_y + max_d, imPr.shape[0]), max(0,c_x - max_d):min(c_x + max_d,imPr.shape[1])]
    cr_im = (cr_im - np.mean(cr_im)) / np.std(cr_im)
    cropped_img = np.stack((cr_im, cr_im, cr_im), axis=-1)
    cropped_img = cv2.resize(cropped_img, (32, 32))

    return cropped_img

def DecodeDet(res, img_shape):
    dets = [[x[-2], int(x[-1]),
             x[0] / 640 * img_shape[0],
             x[1] / 640 * img_shape[1],
             x[2] / 640 * img_shape[0],
             x[3] / 640 * img_shape[1]] for x in res]

    return dets

def folder_files(ImgsPaths):
    folder_files = {}
    for i in range(len(ImgsPaths)):
        folder = ImgsPaths[i][0]
        if folder not in folder_files.keys():
            folder_files[folder] = []
        folder_files[folder].append(i)
    return folder_files

def generate_declaration_file(testing_files, ImgsPaths, InferDets, output, path_to_save):
    Folder_Files = folder_files(ImgsPaths)
    for i in range(len(testing_files)):
        j = 0
        for frame in Folder_Files[testing_files[i]+".arf"]:
            for k in range(len(InferDets[frame][0])):
                InferDets[frame][0][k][4] = output[testing_files[i]]["pred_conf"][j]
                InferDets[frame][0][k][5]  = float(output[testing_files[i]]["pred"][j])
                j+=1
    with open(path_to_save, "wb") as f:
        pickle.dump(ImgsPaths, f)
        pickle.dump(InferDets, f)

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
            
def area(a, b):  
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0
    


def get_day_night(filename, def_day_night): # def_day_night = config["def_day_night"]
    if os.path.splitext(filename)[0].split("_")[0][-4:] in def_day_night['DAY'] :
        day_night = "day"
    elif os.path.splitext(filename)[0].split("_")[0][-4:] in def_day_night['NIGHT']:
        day_night = "night"
    else:
        day_night = 'undefined'
    return day_night

def get_range(filename, def_ranges): # = config["def_ranges"]
    
    for k,v in list(def_ranges.items()):
        if os.path.splitext(filename)[0].split("_")[0][-4:] in str(v):
            return k

################### DATA EXTRACTION METHODS ###################################
def extract_data_gt(training_files, training_sampling, ImgsPaths, InferDets, collection):
 
    if not os.path.exists("./features/target_images"):
        os.mkdir("./features/target_images")
        
    unlabeled = dict()
    unlabeled['Shifted_v'] = dict()
    unlabeled['Shifted_v']['images'] = []

    unlabeled['Shifted_h'] = dict()
    unlabeled['Shifted_h']['images'] = []

    unlabeled['Scaled'] = dict()
    unlabeled['Scaled']['images'] = []

    unlabeled['Blurred'] = dict()
    unlabeled['Blurred']['images'] = []
    
    ul_labels = []
    
    VehicleNameDict= config['cls_id']
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    Folder_Files = folder_files(ImgsPaths)
    
    root = config['data_config']['root']
    normalize_opts = config['normalize']
    
    processed_train = []
    label_train = []  
    features = dict()
    
    FT = []
#    processed_train_fa = []
#    label_train_fa = []
#    contrast_fa = []
    nb_fa = 0
    for i in range(len(training_files)):
        
        print("Preparing data for "+training_files[i]+" ...")
        
        features[training_files[i]] = []
        InputLines = GenInputLines([training_files[i]],AllFrames=False,use_json=True)
        o = arf.arf_open(os.path.join(root, training_files[i]+".arf"))
        for j in range(0, len(InputLines), training_sampling[i]):
            coords_bb = InputLines[j][2]
            fid_= Folder_Files[training_files[i]+".arf"][j]
            res,img_shape=InferDets[fid_]
            dets_=DecodeDet(res,img_shape)
            
            bb_conf = dict()
            bb_overlap = dict()
            for c in range(len(dets_)):
                bb_overlap[c] = []
            
            im = o.load(InputLines[j][1])
            imPr = pre_process(im, config['invert'], config['clip1'], normalize_opts , config['clip2'])
            

            
            nb_gt = -1
            for bb_gt in coords_bb:
                nb_gt += 1
                up_l_x   = bb_gt[0]
                up_l_y   = bb_gt[1]
                bot_r_x      = bb_gt[2]
                bot_r_y      = bb_gt[3]
                c_x = int(bot_r_x - (bot_r_x-up_l_x)//2)
                c_y = int(bot_r_y - (bot_r_y-up_l_y)//2)
                max_d = np.maximum((c_y-up_l_y), (c_x-up_l_x))
                
                cropped_img = crop_patch(imPr, c_x, c_y, max_d)
                processed_train.append(cropped_img)
                label_train.append(0 if bb_gt[4]<2 else 1)
                
                proj_D = {'images': np.transpose(np.array(cropped_img.reshape(1,32,32,3)).astype('float32'), (0,3,1,2)), 'labels':  np.array(0).astype(np.int32).squeeze().reshape((-1))}
                O = test(proj_D, os.path.join(config["data_config"]["projection_model_path"],'best_model.pth'))
                f_128 = O['Fts_test'][1,:,:,:].reshape(1,128)
                unlabeled['Shifted_v']['images'].append(crop_shifted_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y,"v", config["gen_unlabeled"]['shift_v_ratio'] ))
                unlabeled['Shifted_h']['images'].append(crop_shifted_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y,"h", config["gen_unlabeled"]['shift_h_ratio'] ))
                unlabeled['Scaled']['images'].append(crop_expanded_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y, config["gen_unlabeled"]['scale_ratio'] ))
                unlabeled['Blurred']['images'].append(crop_blurred_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y, config["gen_unlabeled"]['blurr_filter']))
                ul_labels.append(0 if bb_gt[4]<2 else 1)
                
                
                rge = get_range(training_files[i], config["def_ranges"])
                time = get_day_night(training_files[i], config["def_day_night"])
                width = 2*(c_x - up_l_x)
                height = 2*(c_y - up_l_y)

               
                
                if label_train[-1] in VehicleNameDict:
                    target_name = VehicleNameDict[label_train[-1]]
                else:
                    target_name = 'OTHER'
                
                if config['save_images']:
                    DrawBoxes(imPr, [bb_gt[:-1]],  training_files[i].split('/')[-1], target_name, 'GT', j, 'Orig')
                    #img.save("./features/target_images/"+training_files[i].split('/')[-1]+'_'+target_name+'_GTbox_f'+str(j)+'_Orig'+'.jpg','JPEG') 
                dr_box =[ [bb_gt[:-1]],  training_files[i].split('/')[-1], target_name, 'GT', j, 'Aug']
                
                features[training_files[i]].append([0, label_train[-1], -1, j, target_name , time, rge, [width,height],-1, 'NaN', collection + '_/'+ training_files[i]]+ list(f_128[0]))
                FT.append([0, label_train[-1], -1, j, target_name , time, rge, [width,height],-1, 'NaN', collection + '_/'+ training_files[i]] + list(f_128[0]))
                    
                aug_idx = len(features[training_files[i]])+1
                aug_cropped_img, aug_list = augment_patch(imPr, up_l_x, up_l_y, c_x, c_y, dr_box )
                ag_l = 0
                for img in aug_cropped_img:
                    processed_train.append(img)
                    label_train.append(0 if bb_gt[4]<2 else 1)
                    proj_D = {'images': np.transpose(np.array(img.reshape(1,32,32,3)).astype('float32'), (0,3,1,2)), 'labels':  np.array(0).astype(np.int32).squeeze().reshape((-1))}
                    O = test(proj_D, os.path.join(config["data_config"]["projection_model_path"],'best_model.pth'))
                    f_128 = O['Fts_test'][1,:,:,:].reshape(1,128)
                    if label_train[-1] in VehicleNameDict:
                        features[training_files[i]].append([0, label_train[-1], -1, j, VehicleNameDict[label_train[-1]], time, rge, [width,height], aug_idx, aug_list[ag_l], collection + '_/'+ training_files[i]]+ list(f_128[0]))
                        FT.append([0, label_train[-1], -1, j, VehicleNameDict[label_train[-1]], time, rge, [width,height], aug_idx, aug_list[ag_l], collection + '_/'+ training_files[i]] +list(f_128[0]))
                        ag_l = ag_l+1
                    else:
                        features[training_files[i]].append([0, label_train[-1], -1, j,'OTHER' , time, rge, [width,height], aug_idx, aug_list[ag_l], collection + '_/'+ training_files[i]]+ list(f_128[0]))
                        
                        FT.append([0, label_train[-1], -1, j,'OTHER' , time, rge, [width,height], aug_idx, aug_idx, aug_list[ag_l], collection + '_/'+ training_files[i]]+ list(f_128[0]))
                        ag_l = ag_l+1
                        
                    
                rb = Rectangle(up_l_x, up_l_y, bot_r_x, bot_r_y)
                nb_det = -1
                for bb in dets_:
                    nb_det += 1
                    up_l_x_yolo   = int(bb[2])
                    up_l_y_yolo    = int(bb[3])
                    bot_r_x_yolo       = bb[4]
                    bot_r_y_yolo       = bb[5]
         
                    ra = Rectangle(up_l_x_yolo , up_l_y_yolo , bot_r_x_yolo , bot_r_y_yolo )  
                    if area(ra, rb)==0:
                        c_x_yolo = int(bot_r_x_yolo - (bot_r_x_yolo-up_l_x_yolo)//2)
                        c_y_yolo = int(bot_r_y_yolo - (bot_r_y_yolo-up_l_y_yolo)//2)
                        max_d_yolo = np.maximum((c_y_yolo-up_l_y_yolo), (c_x_yolo-up_l_x_yolo))
                        cropped_img = crop_patch(imPr, c_x_yolo, c_y_yolo, max_d_yolo)
                        bb_overlap[nb_det].append(cropped_img)
                        bb_conf[nb_det] = bb[0]
                                  
#                generated_BBs, contrast_BBs = generate_3rd_class(im.shape, imPr, max_d, Rectangle, rb)
#                for idx in range(len(generated_BBs)):
#                    bb = generated_BBs[idx]
#                    contrast_fa.append(contrast_BBs[idx])
#                    cropped_img = crop_patch(imPr, bb[0], bb[1], max_d)
#                    processed_train_fa.append(cropped_img)
#                    label_train_fa.append(2)
                    
            for c in range(len(dets_)):
                if len(bb_overlap[c]) == len(coords_bb):
                    nb_fa += 1
                    processed_train.append(bb_overlap[c][0])
                    label_train.append(2)
                    proj_D = {'images': np.transpose(np.array(bb_overlap[c][0].reshape(1,32,32,3)).astype('float32'), (0,3,1,2)), 'labels':  np.array(0).astype(np.int32).squeeze().reshape((-1))}
                    O = test(proj_D, os.path.join(config["data_config"]["projection_model_path"],'best_model.pth'))
                    f_128 = O['Fts_test'][1,:,:,:].reshape(1,128)
                    
                    unlabeled['Shifted_v']['images'].append(crop_shifted_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y,"v", config["gen_unlabeled"]['shift_v_ratio'] ))
                    unlabeled['Shifted_h']['images'].append(crop_shifted_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y,"h", config["gen_unlabeled"]['shift_h_ratio'] ))
                    unlabeled['Scaled']['images'].append(crop_expanded_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y, config["gen_unlabeled"]['scale_ratio'] ))
                    unlabeled['Blurred']['images'].append(crop_blurred_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y, config["gen_unlabeled"]['blurr_filter']))
                    ul_labels.append(2)                  

                    features[training_files[i]].append([1, 2, bb_conf[c], j, 'Non_Target', time, rge, [width,height],-1,  'NaN', collection + '_/'+ training_files[i]]+ list(f_128[0]))
                    FT.append([1, 2, bb_conf[c], j, 'Non_Target', time, rge, [width,height],-1,  'NaN', collection + '_/'+ training_files[i]]+ list(f_128[0]))
                    target_name = 'Non_Target'
                    dr_box =[ [bb_gt[:-1]],  training_files[i].split('/')[-1], target_name, 'GT', j, 'Aug']


                    
                    aug_cropped_img, aug_list = augment_patch(imPr, up_l_x, up_l_y, c_x, c_y, dr_box )
                    aug_idx = len(features[training_files[i]])+1
                    ag_l = 0
                    for img in aug_cropped_img:
                        processed_train.append(img)
                        label_train.append(2)
                        proj_D = {'images': np.transpose(np.array(img.reshape(1,32,32,3)).astype('float32'), (0,3,1,2)), 'labels':  np.array(0).astype(np.int32).squeeze().reshape((-1))}
                        O = test(proj_D, os.path.join(config["data_config"]["projection_model_path"],'best_model.pth'))
                        f_128 = O['Fts_test'][1,:,:,:].reshape(1,128)
                        features[training_files[i]].append([1, 2, bb_conf[c], j,  'Non_Target', time, rge, [width,height],aug_idx, aug_list[ag_l] , collection + '_/'+ training_files[i]]+ list(f_128[0]))
                        FT.append([1, 2, bb_conf[c], j,  'Non_Target', time, rge, [width,height],aug_idx, aug_list[ag_l] , collection + '_/'+ training_files[i]]+ list(f_128[0]))
                        ag_l = ag_l+1
                                    
#    sorted_train_fa = np.argsort(contrast_fa)
#    nb = min(len(np.array(label_train)[(np.array(label_train)==0)]), len(np.array(label_train)[(np.array(label_train)==1)]))-len(np.array(label_train)[(np.array(label_train)==2)])
#    if nb > 0:
#        processed_train += list(np.array(processed_train_fa)[sorted_train_fa[-nb:]])
#        label_train += list(np.array(label_train_fa)[sorted_train_fa[-nb:]])
#        features += [[0, 2, -1] for ii in range(nb)]

    ul_lbl_train  = np.array(ul_labels)
    ul_lbl_train  = ul_lbl_train.reshape((ul_lbl_train.shape[0],1))
    ul_lbl_train = ul_lbl_train.astype(np.int32).squeeze().reshape((-1))
    
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

    u_train_set = split_l_u(unlabeled['Shifted_v'], unlabeled['Shifted_h'], unlabeled['Scaled'], unlabeled['Blurred'], config["gen_unlabeled"]['Selector'], config["gen_unlabeled"]['percentage'])
    u_labels = copy.deepcopy(u_train_set['labels']) 
    u_train_set['labels'] = [np.zeros_like(u_train_set['labels']) - 1]
    u_train_set['labels'] = np.concatenate(u_train_set['labels'], 0)

    return processed_train, label_train, features, np.array(FT), u_train_set, u_labels

def extract_data_yolo(training_files, training_sampling, ImgsPaths, InferDets, collection):
    VehicleNameDict= config['cls_id']
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    Folder_Files = folder_files(ImgsPaths)
    root = config["data_config"]['root']
    normalize_opts = config['normalize']

    unlabeled = dict()
    unlabeled['Shifted_v'] = dict()
    unlabeled['Shifted_v']['images'] = []

    unlabeled['Shifted_h'] = dict()
    unlabeled['Shifted_h']['images'] = []

    unlabeled['Scaled'] = dict()
    unlabeled['Scaled']['images'] = []

    unlabeled['Blurred'] = dict()
    unlabeled['Blurred']['images'] = []
    
    ul_labels = []
    
    processed_train = []
    label_train = []
    features = dict()
    FT = []
#    processed_train_fa = []
#    label_train_fa = []
#    contrast_fa = []
    nb_fa = 0
    for i in range(len(training_files)):
        
        print("Preparing data for "+training_files[i]+" ...")
        
        features[training_files[i]] = []
        InputLines = GenInputLines([training_files[i]],AllFrames=False,use_json=True)
        o = arf.arf_open(os.path.join(root, training_files[i])+'.arf')
        for fid in range(0,len(Folder_Files[training_files[i]+".arf"]), training_sampling[i]):
            coords_gt = InputLines[fid][2]
            fid_ = Folder_Files[training_files[i]+".arf"][fid]
            frame_idx = ImgsPaths[fid_][1]
            res, img_shape = InferDets[fid_]
            dets_ = DecodeDet(res, img_shape)            
            if len(dets_) == 0:
                pass
            else:
                for bb in dets_:
                    im = o.load(frame_idx)
                    imPr = pre_process(im, config['invert'], config['clip1'], normalize_opts, config['clip2'])
                    up_l_x   = int(bb[2])
                    up_l_y   = int(bb[3])
                    bot_r_x      = bb[4]
                    bot_r_y      = bb[5]
                    c_x = int(bot_r_x - (bot_r_x-up_l_x)//2)
                    c_y = int(bot_r_y - (bot_r_y-up_l_y)//2)
                    max_d = np.maximum((c_y-up_l_y), (c_x-up_l_x))
                    cropped_img = crop_patch(imPr, c_x, c_y, max_d)
                    processed_train.append(cropped_img)

                    proj_D = {'images': np.transpose(np.array(cropped_img.reshape(1,32,32,3)).astype('float32'), (0,3,1,2)), 'labels':  np.array(0).astype(np.int32).squeeze().reshape((-1))}
                    O = test(proj_D, os.path.join(config["data_config"]["projection_model_path"],'best_model.pth'))
                    f_128_ = O['Fts_test'][1,:,:,:].reshape(1,128)

                    unlabeled['Shifted_v']['images'].append(crop_shifted_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y,"v", config["gen_unlabeled"]['shift_v_ratio'] ))
                    unlabeled['Shifted_h']['images'].append(crop_shifted_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y,"h", config["gen_unlabeled"]['shift_h_ratio'] ))
                    unlabeled['Scaled']['images'].append(crop_expanded_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y, config["gen_unlabeled"]['scale_ratio'] ))
                    unlabeled['Blurred']['images'].append(crop_blurred_BB_unlabeled(imPr, up_l_x, up_l_y, c_x, c_y, config["gen_unlabeled"]['blurr_filter']))
                    
                    
                    rge = get_range(training_files[i], config["def_ranges"])
                    time = get_day_night(training_files[i], config["def_day_night"])
                    width = 2*(c_x - up_l_x)
                    height = 2*(c_y - up_l_y)
                    
                    

                    ra = Rectangle(up_l_x , up_l_y , bot_r_x , bot_r_y)

                        
                    overlap_bb = dict()
                    for bb_gt in coords_gt:
                        up_l_x_gt   = bb_gt[0]
                        up_l_y_gt   = bb_gt[1]
                        bot_r_x_gt      = bb_gt[2]
                        bot_r_y_gt      = bb_gt[3]
                        c_x_gt = int(bot_r_x_gt - (bot_r_x_gt-up_l_x_gt)//2)
                        c_y_gt = int(bot_r_y_gt - (bot_r_y_gt-up_l_y_gt)//2)
                        max_d = np.maximum((c_y_gt-up_l_y_gt), (c_x_gt-up_l_x_gt))
                        rb = Rectangle(up_l_x_gt , up_l_y_gt , bot_r_x_gt , bot_r_y_gt)
                        overlap_bb[area(ra, rb)] = bb_gt[4]
                    max_area = np.max(list(overlap_bb.keys()))
                    if max_area == 0:
                        nb_fa += 1
                        label_train.append(2)
                        ul_labels.append(2)
                        ####
                        features[training_files[i]].append([1, 2, bb[0], fid, 'Non_Target', time, rge, [width,height],-1, 'NaN', collection + '_/'+ training_files[i]]+ list(f_128_[0]))
                        FT.append([1, 2, bb[0], fid, 'Non_Target', time, rge, [width,height],-1, 'NaN', collection + '_/'+ training_files[i]]+ list(f_128_[0]))
                        target_name = 'Non_Target'
                                                
                        if config['save_images']:
                            DrawBoxes(imPr, [bb[2:]],  training_files[i].split('/')[-1], target_name, 'YOLO', fid, 'Orig')
                            #img.save("./features/target_images/"+training_files[i].split('/')[-1]+'_'+target_name+'_GTbox_f'+str(j)+'_Orig'+'.jpg','JPEG') 
                        
                        ####
                    
                        aug_idx = len(features[training_files[i]])+1 
                        dr_box =[ [bb[2:]],  training_files[i].split('/')[-1], target_name, 'YOLO', fid, 'Aug']  
                        aug_cropped_img, aug_list = augment_patch(imPr, up_l_x, up_l_y, c_x, c_y, dr_box)    
                        ag_l = 0
                                            
                        for img in aug_cropped_img:
                            processed_train.append(img)
                            label_train.append(2)
                            proj_D = {'images': np.transpose(np.array(img.reshape(1,32,32,3)).astype('float32'), (0,3,1,2)), 'labels':  np.array(0).astype(np.int32).squeeze().reshape((-1))}
                            O = test(proj_D, os.path.join(config["data_config"]["projection_model_path"],'best_model.pth'))
                            f_128 = O['Fts_test'][1,:,:,:].reshape(1,128)
                            features[training_files[i]].append([1, 2,  bb[0], fid,  'Non_Target', time, rge, [width,height],aug_idx, aug_list[ag_l], collection + '_/'+ training_files[i]]+ list(f_128[0]))
                            FT.append([1, 2,  bb[0], fid,  'Non_Target', time, rge, [width,height],aug_idx, aug_list[ag_l], collection + '_/'+ training_files[i]]+ list(f_128[0]))
                            ag_l = ag_l+1
                            #features[training_files[i]].append([1, label_train[-1], bb[0], fid])
                        
                    else:
                        label_train.append(0 if overlap_bb[max_area]<2 else 1)
                        ul_labels.append(0 if overlap_bb[max_area]<2 else 1)
                        
                        if label_train[-1] in VehicleNameDict:
                            target_name = VehicleNameDict[label_train[-1]]
                        else:
                            target_name = 'OTHER'
                        
                        if config['save_images']:
                            DrawBoxes(imPr, [bb[2:]],  training_files[i].split('/')[-1], target_name, 'YOLO', fid, 'Orig')
                            #img.save("./features/target_images/"+training_files[i].split('/')[-1]+'_'+target_name+'_GTbox_f'+str(j)+'_Orig'+'.jpg','JPEG') 
                        dr_box =[ [bb[2:]],  training_files[i].split('/')[-1], target_name, 'YOLO', fid, 'Aug']
                        features[training_files[i]].append([0, label_train[-1], -1, fid, target_name , time, rge, [width,height],-1, 'NaN', collection + '_/'+ training_files[i]]+ list(f_128_[0]))
                        FT.append([0, label_train[-1], -1, fid, target_name , time, rge, [width,height],-1, 'NaN', collection + '_/'+ training_files[i]]+ list(f_128_[0]))
                        aug_idx = len(features[training_files[i]])+1
                        aug_cropped_img, aug_list = augment_patch(imPr, up_l_x, up_l_y, c_x, c_y, dr_box )                        

                        #features[training_files[i]].append([1, label_train[-1], bb[0], -1])  
                        ag_l = 0
                        for img in aug_cropped_img:
                            processed_train.append(img)
                            #label_train.append(2)
                            label_train.append(0 if overlap_bb[max_area]<2 else 1)
                            proj_D = {'images': np.transpose(np.array(img.reshape(1,32,32,3)).astype('float32'), (0,3,1,2)), 'labels':  np.array(0).astype(np.int32).squeeze().reshape((-1))}
                            O = test(proj_D, os.path.join(config["data_config"]["projection_model_path"],'best_model.pth'))
                            f_128 = O['Fts_test'][1,:,:,:].reshape(1,128)
                            if label_train[-1] in VehicleNameDict:
                                target_name = VehicleNameDict[label_train[-1]]
                            else:
                                target_name = 'OTHER'
                            features[training_files[i]].append([0, label_train[-1], fid, fid, target_name , time, rge, [width,height],aug_idx, aug_list[ag_l], collection + '_/'+ training_files[i]]+ list(f_128[0]))
                            FT.append([0, label_train[-1], fid, fid, target_name , time, rge, [width,height],aug_idx, aug_list[ag_l], collection + '_/'+ training_files[i]]+ list(f_128[0]))
                            ag_l = ag_l+1
                            #features[training_files[i]].append([1, label_train[-1], bb[0], fid])
                        
#    sorted_train_fa = np.argsort(contrast_fa)
#    nb = min(len(np.array(label_train)[(np.array(label_train)==0)]), len(np.array(label_train)[(np.array(label_train)==1)]))-len(np.array(label_train)[(np.array(label_train)==2)])
#    if nb > 0:
#        processed_train += list(np.array(processed_train_fa)[sorted_train_fa[-nb:]])
#        label_train += list(np.array(label_train_fa)[sorted_train_fa[-nb:]])

    ul_lbl_train  = np.array(ul_labels)
    ul_lbl_train  = ul_lbl_train.reshape((ul_lbl_train.shape[0],1))
    ul_lbl_train = ul_lbl_train.astype(np.int32).squeeze().reshape((-1))
    
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

    u_train_set = split_l_u(unlabeled['Shifted_v'], unlabeled['Shifted_h'], unlabeled['Scaled'], unlabeled['Blurred'], config["gen_unlabeled"]['Selector'], config["gen_unlabeled"]['percentage'])
    u_labels = copy.deepcopy(u_train_set['labels']) 
    u_train_set['labels'] = [np.zeros_like(u_train_set['labels']) - 1]
    u_train_set['labels'] = np.concatenate(u_train_set['labels'], 0)

    return processed_train, label_train, features, np.array(FT), u_train_set, u_labels

def extract_data_both(training_files, training_sampling, ImgsPaths, InferDets, collection):
    processed_train_yolo, label_train_yolo, features_yolo, FT_y, u_train_set_y, u_labels_y = extract_data_yolo(training_files, training_sampling, ImgsPaths, InferDets, collection) 
    processed_train_gt, label_train_gt, features_gt, FT_gt, u_train_set_gt, u_labels_gt = extract_data_gt(training_files, training_sampling, ImgsPaths, InferDets, collection) 
    
    processed_train = processed_train_yolo + processed_train_gt
    label_train = label_train_yolo + label_train_gt
    #u_labels = u_labels_y + u_labels_gt
    features = {**features_yolo, **features_gt}

    u_train_set = dict()
    u_train_set['images'] = np.concatenate((u_train_set_y['images'],u_train_set_gt['images']),axis=0)
    u_train_set['labels'] = np.concatenate((u_train_set_y['labels'],u_train_set_gt['labels']),axis=0)
    u_labels = np.concatenate((u_labels_y, u_labels_gt),axis=0)
    FT = np.concatenate((FT_y, FT_gt),axis=0)
    return processed_train, label_train, features, FT, u_train_set, u_labels

####################### DATA AUGMENTATION METHODS ##########################


        
        
        
def crop_shifted_BB(frame, up_l_x, up_l_y, c_x, c_y, opts, shift_ratio, dr_box):
    # if opt = "v" then vertical shifting, if opt = "h" then horizontal shifting, if opt = "both", the in both directions
    #seed(1)
    #bot_x = up_l_x - 2*(c_x-up_l_x)
    #bot_y = up_l_y - 2*(c_y-up_l_y)
    n = random.uniform(0, 1) - shift_ratio
    wx = np.abs(up_l_x - c_x)
    wy = np.abs(up_l_y - c_y)
    max_d = max(wx,wy)
    dx = np.int(2*n*wx)+1
    dy = np.int(-1*2*n*wy)+1
    
    if opts == "v":
        cr_im = frame[max(c_y - max_d,0):min(c_y + max_d, frame.shape[0]), max(0,c_x - max_d-dx):min(c_x + max_d+dx,frame.shape[1])]
        #bb = [max(c_y - max_d,0), min(c_y + max_d, frame.shape[0]), max(0,c_x - max_d-dx), min(c_x + max_d+dx,frame.shape[1])]
        bb = [max(c_y - max_d,0), min(c_y + max_d, frame.shape[0]), max(0,c_x - max_d-dx), min(c_x + max_d+dx,frame.shape[1])]
        
    if opts == "h":
        cr_im = frame[max(c_y - max_d-dy,0):min(c_y + max_d+dy, frame.shape[0]), max(0,c_x - max_d):min(c_x + max_d,frame.shape[1])]
        bb = [max(c_y - max_d-dy,0), min(c_y + max_d+dy, frame.shape[0]), max(0,c_x - max_d), min(c_x + max_d,frame.shape[1])]

    if opts == "both":
        cr_im = frame[max(c_y - max_d-dy,0):min(c_y + max_d+dy, frame.shape[0]), max(0,c_x - max_d-dx):min(c_x + max_d+dx,frame.shape[1])]
        bb = [max(c_y - max_d-dy,0), min(c_y + max_d+dy, frame.shape[0]), max(0,c_x - max_d-dx), min(c_x + max_d+dx,frame.shape[1])]

    if config['save_images']:
        #frame = (frame-np.mean(frame))/np.std(frame)
        cv2.imwrite(set_img_name( dr_box[1], dr_box[2], dr_box[3], dr_box[4], dr_box[5]+'_Shifted'), cr_im)
        #DrawBoxes(frame, [bb], dr_box[1], dr_box[2], dr_box[3], dr_box[4], dr_box[5]+'_Shifted')
        
    cr_im = (cr_im-np.mean(cr_im))/np.std(cr_im)
    cropped_img = np.stack((cr_im,cr_im,cr_im),axis=-1)
    cropped_img = cv2.resize(cropped_img,(32,32))
    
    return cropped_img

def set_img_name(filename, target_name, GTorYOLO, frameID, aug):
    return "./features/target_images/"+filename+'_'+target_name+'_' + GTorYOLO +'_f'+str(frameID)+'_' + aug +'.jpg'



def crop_blurred_BB(frame, up_l_x, up_l_y, c_x, c_y, filter_param, dr_box):
    #n = random.uniform(0, 2)
    wx = 2*np.abs(up_l_x - c_x)
    wy = 2*np.abs(up_l_y - c_y)
    w = max(wx,wy)

    cr_im = frame[up_l_y-(w-wy)//2:up_l_y-2*(up_l_y-c_y)+(w-wy)//2 , up_l_x-(w-wx)//2:up_l_x+2*(c_x-up_l_x)+(w-wx)//2]
    cr_im = (cr_im-np.mean(cr_im))/np.std(cr_im)
    cr_im = gaussian_filter(cr_im, sigma = filter_param)
    #cr_im = (cr_im-np.mean(cr_im))/np.std(cr_im)
    cropped_img = np.stack((cr_im,cr_im,cr_im),axis=-1)
    cropped_img = cv2.resize(cropped_img,(32,32))

    if config['save_images']:
        cr_im = frame[up_l_y-(w-wy)//2:up_l_y-2*(up_l_y-c_y)+(w-wy)//2 , up_l_x-(w-wx)//2:up_l_x+2*(c_x-up_l_x)+(w-wx)//2]
        cr_im = gaussian_filter(cr_im, sigma = filter_param)
        #frame = (frame-np.mean(frame))/np.std(frame)
        #imPr = gaussian_filter(frame,   sigma = filter_param)
        cv2.imwrite(set_img_name( dr_box[1], dr_box[2], dr_box[3], dr_box[4], dr_box[5]+'_Blurred'), cr_im)
        #DrawBoxes(imPr, dr_box[0], dr_box[1], dr_box[2], dr_box[3], dr_box[4], dr_box[5]+'_Blurred')
    
    return cropped_img

def crop_expanded_BB(frame, up_l_x, up_l_y, c_x, c_y, expand_ratio, dr_box):
    n = random.uniform(0, 1)- expand_ratio
    wx = np.abs(up_l_x - c_x)
    wy = np.abs(up_l_y - c_y)
    dx = np.int(2*n*wx)
    dy = np.int(2*n*wy)
    max_d = max(wx, wy)

    cr_im = frame[max(c_y - max_d-dy,0):min(c_y + max_d+dy, frame.shape[0]), max(0,c_x - max_d-dx):min(c_x + max_d+dx,frame.shape[1])]#frame[up_l_y-(w-wy)//2-dy:up_l_y-2*(up_l_y-c_y)+(w-wy)//2 +dy, up_l_x-(w-wx)//2-dx:up_l_x+2*(c_x-up_l_x)+(w-wx)//2+dx]
    
    if config['save_images']:
        
        #imPr = (frame-np.mean(frame))/np.std(frame)
        #bb = [max(c_y - max_d-dy,0),min(c_y + max_d+dy, frame.shape[0]), max(0,c_x - max_d-dx),min(c_x + max_d+dx,frame.shape[1])]
        #DrawBoxes(frame, [bb], dr_box[1], dr_box[2], dr_box[3], dr_box[4], dr_box[5]+'_Scaled')
        cv2.imwrite(set_img_name( dr_box[1], dr_box[2], dr_box[3], dr_box[4], dr_box[5]+'_Scaled'), cr_im)
        
    cr_im = (cr_im-np.mean(cr_im))/np.std(cr_im)
    cropped_img = np.stack((cr_im,cr_im,cr_im),axis=-1)
    cropped_img = cv2.resize(cropped_img,(32,32))


    return cropped_img

def augment_patch(frame, up_l_x, up_l_y, c_x, c_y, dr_box):
    
    cropped_img = []
    aug_list =[]
    
    if config["data_config"]["aug_config"]['Shift']['select']:
        for i in range(config["data_config"]["aug_config"]['Shift']['K']):
            cropped_img.append(crop_shifted_BB(frame, up_l_x, up_l_y, c_x, c_y, config["data_config"]["aug_config"]['Shift']['axis'],config["data_config"]["aug_config"]['Shift']['shift_ratio'], dr_box))
            aug_list.append('Shifted_'+config["data_config"]["aug_config"]['Shift']['axis'])
    
    if config["data_config"]["aug_config"]['Blurr']['select']:
        for i in range(config["data_config"]["aug_config"]['Blurr']['K']):
            cropped_img.append(crop_blurred_BB(frame, up_l_x, up_l_y, c_x, c_y, config["data_config"]["aug_config"]['Blurr']['filter_param'], dr_box))
            aug_list.append('Blurred')
    
    if config["data_config"]["aug_config"]['Expand']['select']:
        for i in range(config["data_config"]["aug_config"]['Expand']['K']):
            cropped_img.append(crop_expanded_BB(frame, up_l_x, up_l_y, c_x, c_y, config["data_config"]["aug_config"]['Expand']['expand_ratio'], dr_box))
            aug_list.append('Scaled')
    
    return cropped_img, aug_list
    

def crop_blurred_BB_unlabeled(frame, up_l_x, up_l_y, c_x, c_y, blurr_ratio):
    import random    
    #n = random.uniform(0, 2)
    n = blurr_ratio
    #x = config['shared']['input_size']
    x = 32
    wx = 2*np.abs(up_l_x - c_x)
    wy = 2*np.abs(up_l_y - c_y)
    w = max(wx,wy)

    #cr_im = frame[up_l_y:up_l_y-2*(up_l_y-c_y),up_l_x:up_l_x+2*(c_x-up_l_x)]
    cr_im = frame[up_l_y-(w-wy)//2:up_l_y-2*(up_l_y-c_y)+(w-wy)//2 , up_l_x-(w-wx)//2:up_l_x+2*(c_x-up_l_x)+(w-wx)//2]
    cr_im = gaussian_filter(cr_im, sigma = n)
    cr_im = (cr_im-np.mean(cr_im))/np.std(cr_im)
    cropped_img = np.stack((cr_im,cr_im,cr_im),axis=-1)
    cropped_img = cv2.resize(cropped_img,(x,x))
    
    return cropped_img

def crop_expanded_BB_unlabeled(frame, up_l_x, up_l_y, c_x, c_y, scale_ratio):
    import random
    n = random.uniform(0, 1)-scale_ratio
    #x = config['shared']['input_size']
    x = 32
    wx = np.abs(up_l_x - c_x)
    wy = np.abs(up_l_y - c_y)
    dx = np.int(2*n*wx)
    dy = np.int(2*n*wy)
    max_d = max(wx, wy)

    cr_im = frame[max(c_y - max_d-dy,0):min(c_y + max_d+dy, frame.shape[0]), max(0,c_x - max_d-dx):min(c_x + max_d+dx,frame.shape[1])]#frame[up_l_y-(w-wy)//2-dy:up_l_y-2*(up_l_y-c_y)+(w-wy)//2 +dy, up_l_x-(w-wx)//2-dx:up_l_x+2*(c_x-up_l_x)+(w-wx)//2+dx]


    cr_im = (cr_im-np.mean(cr_im))/np.std(cr_im)
    cropped_img = np.stack((cr_im,cr_im,cr_im),axis=-1)
    cropped_img = cv2.resize(cropped_img,(x,x))
    
    return cropped_img


def crop_shifted_BB_unlabeled(frame, up_l_x, up_l_y, c_x, c_y, opts, shift_ratio):
    # if opt = "v" then vertical shifting, if opt = "h" then horizontal shifting, if opt = "both", the in both directions
    #seed(1)
    #bot_x = up_l_x - 2*(c_x-up_l_x)
    #bot_y = up_l_y - 2*(c_y-up_l_y)
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

        
    cr_im = (cr_im-np.mean(cr_im))/np.std(cr_im)
    cropped_img = np.stack((cr_im,cr_im,cr_im),axis=-1)
    cropped_img = cv2.resize(cropped_img,(32,32))
    
    return cropped_img

def split_l_u(data1,data2,data3,data4, select_data, percentage):
    pct_labels_per_cls = (1/np.sum(select_data))*percentage
    # NOTE: this function assume that train_set is shuffled.
    source =[]
    if select_data[0] == 1:
        u0 = get_ul_per_class(data1,pct_labels_per_cls)
        source.append(u0)
    if select_data[1] == 1:
        u1 = get_ul_per_class(data2,pct_labels_per_cls)
        source.append(u1)
    if select_data[2] == 1:
        u2 = get_ul_per_class(data3,pct_labels_per_cls)
        source.append(u2)
    if select_data[3] == 1:
        u3 = get_ul_per_class(data4,pct_labels_per_cls)
        source.append(u3)
        
    #source = [u0,u1,u2,u3]
    result = {}
    for key in source[0]:
        result[key] = np.concatenate([d[key] for d in source])

    return result

def get_ul_per_class(train_set,pct_labels_per_cls):
    
    rng = np.random.RandomState(1)
    indices = rng.permutation(len(train_set["images"]))
    train_set["images"] = train_set["images"][indices]
    train_set["labels"] = train_set["labels"][indices] 
    
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)
    
    u_images = []
    u_labels = []
    lbls = []
    for c in classes:
        
        n_labels_per_cls = int(pct_labels_per_cls*np.count_nonzero(labels == c))
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]

        u_images += [c_images[:n_labels_per_cls]]
        lbls += [c_labels[:n_labels_per_cls]]
        #u_labels += [np.zeros_like(c_labels[:n_labels_per_cls]) - 1] # dammy label
        u_labels = lbls
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0)}
    #lbls =  np.concatenate(lbls, 0)
    return u_train_set






################### TESTING DATA EXTRACTION METHODS ###################################
def extract_test_gt(testing_files, testing_sampling, ImgsPaths, InferDets, collection):
 
    if not os.path.exists("./features/target_images"):
        os.mkdir("./features/target_images")
        

    
    VehicleNameDict= config['cls_id']
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    Folder_Files = folder_files(ImgsPaths)
    
    root = config['data_config']['root']
    normalize_opts = config['normalize']
    
    processed_test = []
    label_test = []  
    features = dict()
    FT = []
#    processed_test_fa = []
#    label_test_fa = []
#    contrast_fa = []
    nb_fa = 0
    for i in range(len(testing_files)):
        
        print("Preparing data for "+testing_files[i]+" ...")
        
        features[testing_files[i]] = []
        InputLines = GenInputLines([testing_files[i]],AllFrames=False,use_json=True)
        o = arf.arf_open(os.path.join(root, testing_files[i]+".arf"))
        for j in range(0, len(InputLines), testing_sampling[i]):
            coords_bb = InputLines[j][2]
            fid_= Folder_Files[testing_files[i]+".arf"][j]
            res,img_shape=InferDets[fid_]
            dets_=DecodeDet(res,img_shape)
            
            bb_conf = dict()
            bb_overlap = dict()
            for c in range(len(dets_)):
                bb_overlap[c] = []
            
            im = o.load(InputLines[j][1])
            imPr = pre_process(im, config['invert'], config['clip1'], normalize_opts , config['clip2'])
            

            
            nb_gt = -1
            for bb_gt in coords_bb:
                nb_gt += 1
                up_l_x   = bb_gt[0]
                up_l_y   = bb_gt[1]
                bot_r_x      = bb_gt[2]
                bot_r_y      = bb_gt[3]
                c_x = int(bot_r_x - (bot_r_x-up_l_x)//2)
                c_y = int(bot_r_y - (bot_r_y-up_l_y)//2)
                max_d = np.maximum((c_y-up_l_y), (c_x-up_l_x))
                
                cropped_img = crop_patch(imPr, c_x, c_y, max_d)
                processed_test.append(cropped_img)
                label_test.append(0 if bb_gt[4]<2 else 1)
                proj_D = {'images': np.transpose(np.array(cropped_img.reshape(1,32,32,3)).astype('float32'), (0,3,1,2)), 'labels':  np.array(0).astype(np.int32).squeeze().reshape((-1))}
                O = test(proj_D, os.path.join(config["data_config"]["projection_model_path"],'best_model.pth'))
                f_128 = O['Fts_test'][1,:,:,:].reshape(1,128)
             
                
                
                rge = get_range(testing_files[i], config["def_ranges"])
                time = get_day_night(testing_files[i], config["def_day_night"])
                width = 2*(c_x - up_l_x)
                height = 2*(c_y - up_l_y)

               
                
                if label_test[-1] in VehicleNameDict:
                    target_name = VehicleNameDict[label_test[-1]]
                else:
                    target_name = 'OTHER'
                
                if config['save_images']:
                    DrawBoxes(imPr, [bb_gt[:-1]],  testing_files[i].split('/')[-1], target_name, 'GT', j, 'Orig')
                    #img.save("./features/target_images/"+testing_files[i].split('/')[-1]+'_'+target_name+'_GTbox_f'+str(j)+'_Orig'+'.jpg','JPEG') 
                
                features[testing_files[i]].append([0, label_test[-1], -1, j, target_name , time, rge, [width,height],-1 , collection + '_/'+ testing_files[i], f_128])
                FT.append([0, label_test[-1], -1, j, target_name , time, rge, [width,height],-1, collection + '_/'+ testing_files[i], f_128])
 
                rb = Rectangle(up_l_x, up_l_y, bot_r_x, bot_r_y)
                nb_det = -1
                for bb in dets_:
                    nb_det += 1
                    up_l_x_yolo   = int(bb[2])
                    up_l_y_yolo    = int(bb[3])
                    bot_r_x_yolo       = bb[4]
                    bot_r_y_yolo       = bb[5]
         
                    ra = Rectangle(up_l_x_yolo , up_l_y_yolo , bot_r_x_yolo , bot_r_y_yolo )  
                    if area(ra, rb)==0:
                        c_x_yolo = int(bot_r_x_yolo - (bot_r_x_yolo-up_l_x_yolo)//2)
                        c_y_yolo = int(bot_r_y_yolo - (bot_r_y_yolo-up_l_y_yolo)//2)
                        max_d_yolo = np.maximum((c_y_yolo-up_l_y_yolo), (c_x_yolo-up_l_x_yolo))
                        cropped_img = crop_patch(imPr, c_x_yolo, c_y_yolo, max_d_yolo)
                        bb_overlap[nb_det].append(cropped_img)
                        bb_conf[nb_det] = bb[0]
                                  

            for c in range(len(dets_)):
                if len(bb_overlap[c]) == len(coords_bb):
                    nb_fa += 1
                    processed_test.append(bb_overlap[c][0])
                    label_test.append(2)
                    proj_D = {'images': np.transpose(np.array(bb_overlap[c][0].reshape(1,32,32,3)).astype('float32'), (0,3,1,2)), 'labels':  np.array(0).astype(np.int32).squeeze().reshape((-1))}
                    O = test(proj_D, os.path.join(config["data_config"]["projection_model_path"],'best_model.pth'))
                    f_128 = O['Fts_test'][1,:,:,:].reshape(1,128)

                    features[testing_files[i]].append([1, 2, bb_conf[c], j, 'Non_Target', time, rge, [width,height],-1 , collection + '_/'+ testing_files[i], f_128])
                    FT.append([1, 2, bb_conf[c], j, 'Non_Target', time, rge, [width,height],-1, collection + '_/'+ testing_files[i], f_128])
                    target_name = 'Non_Target'
                    dr_box =[ [bb_gt[:-1]],  testing_files[i].split('/')[-1], target_name, 'GT', j, 'Aug']

    return processed_test, label_test, features, np.array(FT)

def extract_test_yolo(testing_files, testing_sampling, ImgsPaths, InferDets, collection):
    VehicleNameDict= config['cls_id']
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    Folder_Files = folder_files(ImgsPaths)
    root = config["data_config"]['root']
    normalize_opts = config['normalize']

    
    processed_test = []
    label_test = []
    features = dict()
    FT = []

    nb_fa = 0
    for i in range(len(testing_files)):
        
        print("Preparing data for "+testing_files[i]+" ...")
        
        features[testing_files[i]] = []
        InputLines = GenInputLines([testing_files[i]],AllFrames=False,use_json=True)
        o = arf.arf_open(os.path.join(root, testing_files[i])+'.arf')
        for fid in range(0,len(Folder_Files[testing_files[i]+".arf"]), testing_sampling[i]):
            coords_gt = InputLines[fid][2]
            fid_ = Folder_Files[testing_files[i]+".arf"][fid]
            frame_idx = ImgsPaths[fid_][1]
            res, img_shape = InferDets[fid_]
            dets_ = DecodeDet(res, img_shape)            
            if len(dets_) == 0:
                pass
            else:
                for bb in dets_:
                    im = o.load(frame_idx)
                    imPr = pre_process(im, config['invert'], config['clip1'], normalize_opts, config['clip2'])
                    up_l_x   = int(bb[2])
                    up_l_y   = int(bb[3])
                    bot_r_x      = bb[4]
                    bot_r_y      = bb[5]
                    c_x = int(bot_r_x - (bot_r_x-up_l_x)//2)
                    c_y = int(bot_r_y - (bot_r_y-up_l_y)//2)
                    max_d = np.maximum((c_y-up_l_y), (c_x-up_l_x))
                    cropped_img = crop_patch(imPr, c_x, c_y, max_d)
                    processed_test.append(cropped_img)    
                    proj_D = {'images': np.transpose(np.array(cropped_img.reshape(1,32,32,3)).astype('float32'), (0,3,1,2)), 'labels':  np.array(0).astype(np.int32).squeeze().reshape((-1))}
                    O = test(proj_D, os.path.join(config["data_config"]["projection_model_path"],'best_model.pth'))
                    f_128 = O['Fts_test'][1,:,:,:].reshape(1,128)
                    
                    rge = get_range(testing_files[i], config["def_ranges"])
                    time = get_day_night(testing_files[i], config["def_day_night"])
                    width = 2*(c_x - up_l_x)
                    height = 2*(c_y - up_l_y)
                    
                    

                    ra = Rectangle(up_l_x , up_l_y , bot_r_x , bot_r_y)

                        
                    overlap_bb = dict()
                    for bb_gt in coords_gt:
                        up_l_x_gt   = bb_gt[0]
                        up_l_y_gt   = bb_gt[1]
                        bot_r_x_gt      = bb_gt[2]
                        bot_r_y_gt      = bb_gt[3]
                        c_x_gt = int(bot_r_x_gt - (bot_r_x_gt-up_l_x_gt)//2)
                        c_y_gt = int(bot_r_y_gt - (bot_r_y_gt-up_l_y_gt)//2)
                        max_d = np.maximum((c_y_gt-up_l_y_gt), (c_x_gt-up_l_x_gt))
                        rb = Rectangle(up_l_x_gt , up_l_y_gt , bot_r_x_gt , bot_r_y_gt)
                        overlap_bb[area(ra, rb)] = bb_gt[4]
                    max_area = np.max(list(overlap_bb.keys()))
                    if max_area == 0:
                        nb_fa += 1
                        label_test.append(2)
                        ####
                        features[testing_files[i]].append([1, 2, bb[0], fid, 'Non_Target', time, rge, [width,height],-1 , collection + '_/'+ testing_files[i], f_128])
                        FT.append([1, 2, bb[0], fid, 'Non_Target', time, rge, [width,height],-1 , collection + '_/'+ testing_files[i], f_128])
                        target_name = 'Non_Target'
                                                
                        if config['save_images']:
                            DrawBoxes(imPr, [bb[2:]],  testing_files[i].split('/')[-1], target_name, 'YOLO', fid, 'Orig')
                            #img.save("./features/target_images/"+testing_files[i].split('/')[-1]+'_'+target_name+'_GTbox_f'+str(j)+'_Orig'+'.jpg','JPEG') 
                        

                        
                    else:
                        label_test.append(0 if overlap_bb[max_area]<2 else 1)
                        
                        if label_test[-1] in VehicleNameDict:
                            target_name = VehicleNameDict[label_test[-1]]
                        else:
                            target_name = 'OTHER'
                        
                        if config['save_images']:
                            DrawBoxes(imPr, [bb[2:]],  testing_files[i].split('/')[-1], target_name, 'YOLO', fid, 'Orig')
                            #img.save("./features/target_images/"+testing_files[i].split('/')[-1]+'_'+target_name+'_GTbox_f'+str(j)+'_Orig'+'.jpg','JPEG') 
                        features[testing_files[i]].append([0, label_test[-1], -1, fid, target_name , time, rge, [width,height],-1 , collection + '_/'+ testing_files[i], f_128])
                        FT.append([0, label_test[-1], -1, fid, target_name , time, rge, [width,height],-1, collection + '_/'+ testing_files[i], f_128])


    return processed_test, label_test, features, np.array(FT)

def extract_test_both(testing_files, testing_sampling, ImgsPaths, InferDets, collection):
    processed_test_yolo, label_test_yolo, features_yolo, FT_y = extract_test_yolo(testing_files, testing_sampling, ImgsPaths, InferDets, collection) 
    processed_test_gt, label_test_gt, features_gt, FT_gt = extract_test_gt(testing_files, testing_sampling, ImgsPaths, InferDets, collection) 
    
    processed_test = processed_test_yolo + processed_test_gt
    label_test = label_test_yolo + label_test_gt
    #u_labels = u_labels_y + u_labels_gt
    features = {**features_yolo, **features_gt}
    FT = np.concatenate((FT_y, FT_gt),axis=0)



    return processed_test, label_test, features, FT
    
    
    
    
    
    
    
    
    
    
    
    
    