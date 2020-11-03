# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:19:44 2019

@author: Fadoua Khmaissia
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse, math, time, json, os
from config import config
from utils.lib import wrn,transform
#from utils.train_config import tr_config
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from utils.prep_unlabeled import *
from datetime import datetime
from IPython import get_ipython
from collections import Counter
import json
def map_classes(train_set, num_cls):
    ClassMapping = config['ClassMapping']
    if num_cls == False:

        data = dict()
        data['images'] = np.concatenate([train_set["images"][train_set["labels"] == c,:,:,:] for c in list(ClassMapping.keys())], axis=0)
        data['labels'] = np.concatenate([train_set["labels"][train_set["labels"] == c] for c in list(ClassMapping.keys())], axis=0)
        data['labels'] = np.array([ClassMapping[data['labels'][idx]] for idx in range(len(data['labels']))]).astype(np.int32)
        Train_d_set = data
    elif num_cls == True:
        data = dict()
        data['images'] = np.concatenate([train_set["images"][train_set["labels"] == c,:,:,:] for c in list(ClassMapping.keys())], axis=0)
        data['labels'] = np.concatenate([train_set["labels"][train_set["labels"] == c] for c in list(ClassMapping.keys())], axis=0)
        data['labels'] = np.array([ClassMapping[data['labels'][idx]] for idx in range(len(data['labels']))])
        bal_num = min([Counter(list(data["labels"]))[k] for k in Counter(list(data["labels"])).keys()])
        data3 = dict()
        data3['images'] = np.concatenate([train_set["images"][train_set["labels"] == c,:,:,:] for c in ['CLUTTER_y', 'CLUTTER', 'CLUTTER+']], axis=0)
        if data3['images'].shape[0] > bal_num:
            data3['images'] = data3['images'][:bal_num,:,:,:]
        data3['labels'] = np.zeros(len(data3['images']), dtype=int) +2 #np.zeros_like(u_train_set['labels']) - 1
        
        Train_d_set = dict()
        Train_d_set['images'] = np.concatenate([data['images'],data3['images']],axis =0)
        Train_d_set['labels'] = np.concatenate([data['labels'],data3['labels']],axis =0)
            
        
    else:
        print('Enter a valid number of classes. This program accepts 2 or 3 classes\n')
    return Train_d_set

def training_code(alg, l_train_set, val_set, u_train_set, num_cls, alg_id):
    if not os.path.exists(config["model_path"]):
        os.mkdir(config["model_path"])
    # if not os.path.exists(os.path.join(config["model_path"],'plots')):
    #     os.mkdir(os.path.join(config["model_path"],'plots'))
    if not os.path.exists(os.path.join(config["model_path"],alg_id)):
        os.mkdir(os.path.join(config["model_path"],alg_id))
    if not os.path.exists(os.path.join(config["model_path"],alg_id,'plots')):
        os.mkdir(os.path.join(config["model_path"],alg_id,'plots'))

    ClassMapping = config['ClassMapping']
    with open(os.path.join(config["model_path"],alg_id, 'log.txt'), 'w') as file:
        file.write('Start time = '+datetime.now().strftime("%m/%d/%Y , %H:%M:%S")+'\n')
        file.write('Training algorithm = '+alg + ', Include Non Target Class = ' + str(num_cls) + '\n\n')
        file.write('Training targets distribution: ' + json.dumps(Counter(list(l_train_set["labels"])))+'\n')
        file.write('Validation targets distribution: ' + json.dumps(Counter(list(val_set["labels"])))+'\n\n')
    # _DATA_DIR = './features/'+'/CNN_data'
    # u_train_set = np.load(os.path.join(_DATA_DIR,"u_train_set.npy"), allow_pickle=True).item()
    # l_train_set = np.load(os.path.join(_DATA_DIR,"train.npy"), allow_pickle=True).item()
    # val_set = np.load(os.path.join(_DATA_DIR,"val.npy"), allow_pickle=True).item()

    # u_train_set = split_l_u(u_train_set['Shifted_v'], u_train_set['Shifted_h'], u_train_set['Scaled'], u_train_set['Blurred'], [1,1,0,1], 0.5)
    # u_train_set['labels'] = np.zeros_like(u_train_set['labels']) - 1
    l_train_set = map_classes(l_train_set, num_cls)
    val_set = map_classes(val_set, num_cls)
    

    
    print()
    TrAcc = np.array(0)
    ValAcc=np.array(0)
    valIter=np.array(0)
    SupL=np.array(0)
    SemSL=np.array(0)

    
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    
    condition = {}
    exp_name = ""
    
    print("dataset : {}".format("data"))
    condition["dataset"] = "data"
    exp_name += str("data") + "_"
    tr_config = config['train_config']
    dataset_cfg = {}
    transform_fn = transform.transform(*tr_config["transform"]) # transform function (flip, crop, noise)
    # dataset_cfg["num_classes"] = num_cls
    if num_cls:
        dataset_cfg["num_classes"] = len(set(ClassMapping.values())) + 1
    else:
        dataset_cfg["num_classes"] = len(set(ClassMapping.values()))
    l_train_dataset = tr_config["dataset"](l_train_set)
    u_train_dataset = tr_config["dataset"](u_train_set)
    val_dataset = tr_config["dataset"](val_set)
    #test_dataset = dataset_cfg["dataset"](test_set)
    
    print("labeled data : {}, unlabeled data : {}, training data : {}".format(
        len(l_train_dataset), len(u_train_dataset), len(l_train_dataset)+len(u_train_dataset)))
    print("validation data : {}".format(len(val_dataset)))
    condition["number_of_data"] = {
        "labeled":len(l_train_dataset), "unlabeled":len(u_train_dataset),
        "validation":len(val_dataset), #"test":len(test_dataset)
    }
    
    class RandomSampler(torch.utils.data.Sampler):
        """ sampling without replacement """
        def __init__(self, num_data, num_sample):
            iterations = num_sample // num_data + 1
            self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]
    
        def __iter__(self):
            return iter(self.indices)
    
        def __len__(self):
            return len(self.indices)
    
    shared_cfg = tr_config#["shared"]
    if alg != "supervised":
        # batch size = 0.5 x batch size
        l_loader = DataLoader(
            l_train_dataset, shared_cfg["batch_size"]//2, drop_last=True,
            sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]//2)
        )
    else:
        l_loader = DataLoader(
            l_train_dataset, shared_cfg["batch_size"], drop_last=True,
            sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"])
        )
    print("algorithm : {}".format(alg))
    condition["algorithm"] = alg
    exp_name += str(alg) + "_"
    
    u_loader = DataLoader(
        u_train_dataset, shared_cfg["batch_size"]//2, drop_last=True,
        sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]//2)# 
    )
    
    val_loader = DataLoader(val_dataset, 128, shuffle=False, drop_last=False)
    l_tr_loader = DataLoader(l_train_dataset, 128, shuffle=False, drop_last=False)
    #test_loader = DataLoader(test_dataset, 128, shuffle=False, drop_last=False)
    
   # u_tr_loader = DataLoader(u_train_dataset, 128, shuffle=False, drop_last=False)
    
    print("maximum iteration : {}".format(min(len(l_loader), len(u_loader))))
    train_step = max(len(u_train_dataset),len(l_train_dataset))//shared_cfg["batch_size"] + 1
    alg_cfg = tr_config[alg]
    print("parameters : ", alg_cfg)
    condition["h_parameters"] = alg_cfg
    
    if 0 > 0:
        print("entropy maximization : {}".format(0))
        exp_name += "em_"
    condition["entropy_maximization"] = 0
    print(dataset_cfg["num_classes"])
    #model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)

    model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
    #model.load_state_dict(torch.load(r".\best_model.pth"))
    #model.output = nn.Linear(model.output.in_features, dataset_cfg["num_classes"])
    #model = model.to(device)    

    optimizer = optim.Adam(model.parameters(), lr=alg_cfg["lr"])
    
    trainable_paramters = sum([p.data.nelement() for p in model.parameters()])
    print("trainable parameters : {}".format(trainable_paramters))
    
    if alg == "VAT": # virtual adversarial training
        from lib.algs.vat import VAT
        ssl_obj = VAT(alg_cfg["eps"]["data"], alg_cfg["xi"], 1)
    elif alg == "PL": # pseudo label
        from lib.algs.pseudo_label import PL
        ssl_obj = PL(alg_cfg["threashold"])
    elif alg == "MT": # mean teacher
        from lib.algs.mean_teacher import MT
        t_model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
        t_model.load_state_dict(model.state_dict())
        ssl_obj = MT(t_model, alg_cfg["ema_factor"])
    elif alg == "PI": # PI Model
        from lib.algs.pimodel import PiModel
        ssl_obj = PiModel()
    elif alg == "ICT": # interpolation consistency training
        from utils.lib.algs.ict import ICT
        t_model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
        t_model.load_state_dict(model.state_dict())
        ssl_obj = ICT(alg_cfg["alpha"], t_model, alg_cfg["ema_factor"])
    elif alg == "MM": # MixMatch
        from utils.lib.algs.mixmatch import MixMatch
        ssl_obj = MixMatch(alg_cfg["T"], alg_cfg["K"], alg_cfg["alpha"])
    elif alg == "supervised":
        pass
    else:
        raise ValueError("{} is unknown algorithm".format(alg))
    
    print()
    iteration = 0
    ttt=0
    maximum_val_acc = 0
    s = time.time()
    for l_data, u_data in zip(l_loader, u_loader):
        iteration += 1
        l_input, target = l_data
        l_input, target = l_input.to(device).float(), target.to(device).long()
    
        if alg != "supervised" and alg != "MM": # for ssl algorithm
            u_input, dummy_target = u_data
            u_input, dummy_target = u_input.to(device).float(), dummy_target.to(device).long()
    
            target = torch.cat([target, dummy_target], 0)
            unlabeled_mask = (target == -1).float()
    
            inputs = torch.cat([l_input, u_input], 0)
            [outputs, feats] = model(inputs)
    
            # ramp up exp(-5(1 - t)^2)
            coef = alg_cfg["consis_coef"] * math.exp(-5 * (1 - min(iteration/shared_cfg["warmup"], 1))**2)
            ssl_loss = ssl_obj(inputs, outputs.detach(), model, unlabeled_mask) * coef
            cls_loss = F.cross_entropy(outputs, target, reduction="none", ignore_index=-1).mean()
        
        elif alg == "MM":
            u_input, dummy_target = u_data
            u_input, dummy_target = u_input.to(device).float(), dummy_target.to(device).long()
    
            target = torch.cat([target, dummy_target], 0)
            unlabeled_mask = (target == -1).float()
    
            inputs = torch.cat([l_input, u_input], 0)
            [outputs, feats] = model(inputs)
            coef = alg_cfg["consis_coef"] * math.exp(-5 * (1 - min(iteration/shared_cfg["warmup"], 1))**2)
            cls_loss, ssl_loss, y_hat = ssl_obj(inputs, target, model, unlabeled_mask, dataset_cfg["num_classes"])
            #cls_loss = torch.zeros(1).to(device)
            #print(ssl_loss)
    
        else:
            [outputs, feats] = model(l_input)
            coef = 0
            ssl_loss = torch.zeros(1).to(device)
            cls_loss = F.cross_entropy(outputs, target, reduction="none", ignore_index=-1).mean()
    
        # supervised loss
        
        #cls_loss = 0
        
        loss = cls_loss + ssl_loss
        
        #loss = ssl_loss
        
        #print(loss)
    
        if 0 > 0:
            loss -= 0 * ((outputs.softmax(1) * F.log_softmax(outputs, 1)).sum(1) * unlabeled_mask).mean()
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if alg == "MT" or alg == "ICT":
            # parameter update with exponential moving average
            ssl_obj.moving_average(model.parameters())
        # display
        if iteration == 1 or (iteration % 1000) == 0:
            wasted_time = time.time() - s
    
            rest = (shared_cfg["iteration"] - iteration)/100 * wasted_time / 60
            print("iteration [{}/{}] cls loss : {:.6e}, SSL loss : {:.6e}, coef : {:.5e}, time : {:.3f} iter/sec, rest : {:.3f} min, lr : {}".format(
                iteration, shared_cfg["iteration"], cls_loss.item(), ssl_loss.item(), coef, 100 / wasted_time, rest, optimizer.param_groups[0]["lr"]),
                "\r", end="")
            s = time.time()
    
        # validation
        #if (iteration % 500) == 0 or iteration == shared_cfg["iteration"]:
        if (iteration % 500) == 0 or iteration == shared_cfg["iteration"]:
            with torch.no_grad():
                model.eval()
                print()
                print("### validation ###")
                sum_acc = 0
                s = time.time()
                Fts_val=np.empty([1,128,1,1])
                for j, data in enumerate(val_loader):
                    input, target = data
                    input, target = input.to(device).float(), target.to(device).long()
    
                    [output, ftv] = model(input)
                    Fts_val=np.concatenate((Fts_val,ftv.cpu().detach().numpy()),0)
    
                    pred_label = output.max(1)[1]
                    sum_acc += (pred_label == target).float().sum()
                    if ((j+1) % 10) == 0:
                        d_p_s = 10/(time.time()-s)
                        print("[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
                            j+1, len(val_loader), d_p_s, (len(val_loader) - j-1)/d_p_s
                        ), "\r", end="")
                        s = time.time()
                acc = sum_acc/float(len(val_dataset))
                
                sum_acc_tr = 0.
                for j, data in enumerate(l_tr_loader):
                    input, target = data
                    input, target = input.to(device).float(), target.to(device).long()
    
                    [output, _] = model(input)
    
                    pred_label = output.max(1)[1]
                    sum_acc_tr += (pred_label == target).float().sum()
                acc_tr = sum_acc_tr/float(len(l_train_dataset))

                print()
                print("Validation accuracy : {}".format(acc))
                print("Training accuracy : {}".format(acc_tr))

                SupL=np.append(SupL,cls_loss.item()*100)
                SemSL=np.append(SemSL,ssl_loss.item()*100)
                ValAcc=np.append(ValAcc,acc.item()*100)
                valIter=np.append(valIter,iteration)
                TrAcc = np.append(TrAcc,acc_tr.item()*100)
                
                
                import matplotlib.pyplot as plt
                from pylab import rcParams
                rcParams['figure.figsize'] = 10, 5 #sets figure size
                #get_ipython().run_line_magic('matplotlib', 'inline')
                plt.plot(TrAcc[1:] , 'r')
                plt.plot(ValAcc[1:], 'b')
                plt.xticks(range(len(valIter)), ["%.0f" % x for x in valIter/100])
                plt.title('Accuracy evolution across iterations')
                plt.legend(['Training','Validation'])
                plt.xlabel('Iter/100')
                plt.ylabel('Accuracy')
                plt.grid()
                plt.savefig(os.path.join(os.path.join(config["model_path"],alg_id, 'plots'), "Accuracy_Plot - Iter"+str(iteration)+".png"))
                #plt.show()
                # test
                if (maximum_val_acc <acc):
                    
                   
                    maximum_val_acc = acc
                    sum_acc = 0.
                    s = time.time()
#                    confusion_matrix = torch.zeros(dataset_cfg["num_classes"],dataset_cfg["num_classes"])
#                    pr=torch.tensor((), dtype=target.dtype,device=target.device)
#                    lab=torch.tensor((), dtype=target.dtype,device=target.device)
    #                    Fts=np.empty([1,128,1,1])
    #                    SupL=np.append(SupL,cls_loss.item()*100)
    #                    SemSL=np.append(SemSL,ssl_loss.item()*100)
    #                    ValAcc=np.append(ValAcc,acc.item()*100)
    #                    valIter=np.append(valIter,iteration)
    # =============================================================================
# =============================================================================
#                    if alg == "MM":
#                        u_lab = torch.tensor((), dtype=target.dtype,device=target.device)
#    
#                        for j, data in enumerate(u_tr_loader):
#                            u, dumm = data
#                            u, dumm = u.to(device).float(), dumm.to(device).long()
#        
#                            target_u = torch.cat([target, dumm], 0)
#                            inps = torch.cat([input, u], 0)
#        
#                            unlabeled_mask = (target_u == -1).float()
#                            #[output, feat] = model(inputs)
#                            cls_loss, ssl_loss, y_hat = ssl_obj(inps, target_u, model, unlabeled_mask, dataset_cfg["num_classes"])
#                            u_lab=torch.cat((u_lab, y_hat.max(1)[1][:int(y_hat.shape[0]/2)]), 0)
#                        ul_acc = accuracy_score(ul, u_lab.cpu().detach().numpy())
#                        U.append(ul_acc)
#                
                    #sum_acc += (y_hat.max(1)[1][:50] == target).float().sum()

    
# =============================================================================

#                     TrAcc = np.append(TrAcc,train_acc.item()*100)
# 

# =============================================================================
                    #print("training accuracy : {}".format(train_acc))

    
                    torch.save(model.state_dict(), os.path.join(config["model_path"],alg_id, datetime.now().strftime("%m-%d-%Y_%H%M%S_")+alg+"_model_Vacc_"+str(maximum_val_acc.cpu().detach().numpy()*100)[:5]+'.pth'))
                    #torch.save(model.state_dict(), os.path.join(args.output, "best_model.pth"))
                    #torch.save(model, os.path.join(args.output, "model.pth"))
#                    tr_features = Fts
                    val_features = Fts_val
                    
                    #ttt=train_acc

    
    
            model.train()
            s = time.time()
        # lr decay
        if iteration == shared_cfg["lr_decay_iter"]:
            optimizer.param_groups[0]["lr"] *= shared_cfg["lr_decay_factor"]
    
# =============================================================================
#     print("train acc : {}".format(train_acc))
#     condition["train_acc"] = ttt.item()
#     
#     exp_name += str(int(time.time())) # unique ID
#     if not os.path.exists(args.output):
#         os.mkdir(args.output)
#     with open(os.path.join(args.output, exp_name + ".json"), "w") as f:
#         json.dump(condition, f)
# =============================================================================
        
    outpts = dict()
    outpts ['val_accuracy'] = maximum_val_acc.cpu().detach().numpy()
    #outpts ['model'] = best_mdl
    outpts ['SupL'] = SupL
    outpts ['SemSL'] = SemSL #outpts ['tr_features'] = tr_features
    outpts ['val_features'] = val_features
    outpts['TrAccs'] = TrAcc
    outpts['ValAccs'] = ValAcc
    outpts['iter'] = valIter

    Oi = dict()
    for k,v in outpts.items():
        Oi[k] =  np.array(v,dtype=float).squeeze()
    Oi['val_features'] =  np.array(-1,dtype=float).squeeze()
    
    with open(os.path.join(config["model_path"],alg_id, 'log.txt'), 'a') as file:
        file.write('Training outputs: \n' + str(Oi)+'\n\n')    
        file.write('End time = '+datetime.now().strftime("%m/%d/%Y , %H:%M:%S")+'\n')
    #np.save(outpts, os.path.join(config["model_path"], alg_id+ "training_outputs"))


#    with open(config["model_path"]+'/Fold#%d_Val%d/train_results_[%.4f].pickle'%(k, val_rng, maximum_val_acc), 'wb') as handle:
 #       pickle.dump(outpts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #return outpts

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    
    import numpy as np
    import itertools
    import matplotlib.pyplot as plt

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    


####
def test(test_set,model_path):
   
    TsAcc = np.array(0)

    
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    dataset_cfg = tr_config['data']
    transform_fn = transform.transform(*dataset_cfg["transform"])
    model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
    model.load_state_dict(torch.load(model_path)) 
    dataset_cfg = tr_config['data']
    test_dataset = dataset_cfg["dataset"](test_set)
    test_loader = DataLoader(test_dataset, 128, shuffle=False, drop_last=False)
    
    with torch.no_grad():
        model.eval()
        print("### test ###")
        sum_acc = 0.

        confusion_matrix = torch.zeros(dataset_cfg["num_classes"],dataset_cfg["num_classes"])
        pr=torch.tensor((), dtype=torch.int64,device=device)
        lab=torch.tensor((), dtype=torch.int64,device=device)
        Fts_test=np.empty([1,128,1,1])
        #conf=np.empty([1,10])
        conf = torch.tensor((), dtype=torch.float32,device=device)
        for j, data in enumerate(test_loader):
            input, target = data
            input, target = input.to(device).float(), target.to(device).long()
            [output, ft] = model(input)
            Fts_test=np.concatenate((Fts_test,ft.cpu().detach().numpy()),0)
            #conf = np.concatenate((conf,output.cpu().detach().numpy()),0)
            conf=torch.cat((conf, output), 0)
            pred_label = output.max(1)[1]
            pr=torch.cat((pr, pred_label), 0)
            lab=torch.cat((lab, target), 0)
            sum_acc += (pred_label == target).float().sum()
            for tt, pp in zip(target.view(-1), pred_label.view(-1)):
                confusion_matrix[tt.long(), pp.long()] += 1
            #if ((j+1) % 10) == 0:
            #   d_p_s = 100/(time.time()-s)
            #    print("[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
            #           j+1, len(test_loader), d_p_s, (len(test_loader) - j-1)/d_p_s
            #   ), "\r", end="")
            #    s = time.time()
            
        test_acc = sum_acc / float(len(test_dataset))
        TsAcc = np.append(TsAcc,test_acc.item()*100)
        print("test accuracy : {}".format(test_acc))  

    
    outpts = dict()
    outpts ['test_accuracy'] = test_acc
    outpts ['conf_mat'] = confusion_matrix
    outpts ['targets'] = lab
    outpts ['preds'] = pr
    outpts['TsAccs'] = TsAcc
    outpts['Fts_test']=Fts_test
    #outpts['conf']=np.delete(conf,0,0)
    outpts['conf']=conf
    return outpts