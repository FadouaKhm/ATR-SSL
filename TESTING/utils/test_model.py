import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from utils import wrn, prep_data, transform
from config import config 

def test(test_set, model_path, NT_class):
    if NT_class=="True":
        num_classes = len(np.unique(list(config["ClassMapping"].values())))+1
    else:
        num_classes = len(np.unique(list(config["ClassMapping"].values())))

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
        
    transform_fn = transform.transform(*config['data_config']["transform"])
    model = wrn.WRN(2, num_classes, transform_fn).to(device)
    model.load_state_dict(torch.load(model_path)) 
    test_dataset = prep_data.data(test_set)
    test_loader = DataLoader(test_dataset, config['data_config']["batch_size"], shuffle=False, drop_last=False)
    
    with torch.no_grad():
        model.eval()
        pred = []
        pred_conf = []
        for j, data in enumerate(test_loader):
            input, target = data
            input, target = input.to(device).float(), target.to(device).long()
            [output, ft]  = model(input)
            if NT_class=="False":
                try:
                    confs = F.softmax(output)
                    for i in range(input.shape[0]):
                        pred.append(np.argmax(confs[i].cpu().numpy()))
                        pred_conf.append(confs[i].cpu().numpy())
    
                except:
                    confs = F.softmax(output)
                    pred.append(np.argmax(confs.cpu().numpy()))
                    pred_conf.append(confs.cpu().numpy())
            else:
                try:
                    confs = F.softmax(output)
                    for i in range(input.shape[0]):
                        pred.append(np.argmax(confs[i].cpu().numpy()[:-1]))
                        pred_conf.append(confs[i].cpu().numpy()[pred[-1]])
                except:
                    confs = F.softmax(output)
                    pred.append(np.argmax(confs.cpu().numpy()[:-1]))
                    pred_conf.append(confs.cpu().numpy()[pred[-1]])
                

    output = dict()
    output ['pred'] = pred
    output["pred_conf"] = pred_conf
    return output


