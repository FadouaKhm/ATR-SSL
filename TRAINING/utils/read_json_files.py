from utils.arf import arf_open
import os
from collections import OrderedDict
from config import config
from utils.atlas_scorer.score import Scorer
from dateutil.parser import parse
import datetime

def GenInputFileList(FileTS):
    InputFileList=[]
    for ifel in FileTS:
        if ifel[-4:]=='.txt':
            with open(ifel) as f:
                extra_lines=f.read().splitlines()
                InputFileList.extend(extra_lines)
        else:
            InputFileList.append(ifel)

    return InputFileList

def get_time(start_time, end_time):
    start_time = start_time.time()
    end_time = end_time.time()
    if start_time >= datetime.time(6,30,0) and end_time >= datetime.time(6,30,0) and end_time <= datetime.time(17,0,0) and start_time <= datetime.time(17,0,0):
        return 'day'
    else:
        return 'night'

def GenInputLines(InputFileSet, OptRange, ylim__max, xlim_max):
    InputFileList = []
    FLSampRate = []
    scorer = Scorer()

    DataRoot = config["data_config"]["root"]
    TargetRangeLimit = 3500
    #EntityNameDict = dict(zip(EntityList,range(len(EntityList))))

    EntityNameDict = list(config['ClassMapping'].keys())
    for ifel in InputFileSet:
        if ',' in ifel:
            NSR=int(ifel[ifel.find(',')+1:])
            ifel=ifel[:ifel.find(',')]
        else:
            NSR=1
        InputFileList.append(ifel)
        FLSampRate.append(NSR)
    InputLines = []
    for FileName,NSR in zip(InputFileList,FLSampRate):
        Ann = OrderedDict()
        aspects = OrderedDict()
        ranges = OrderedDict()
        time_ = OrderedDict()
        truth_file = os.path.join(DataRoot, FileName.replace('/arf/','/json_truth/')+'.truth.json')
        truth_obj = scorer.load_truth_file(truth_file,return_dict=False)
        df_Labels = scorer.annotation_objects_to_table_expanded(truth_obj)
        df_Labels = df_Labels.rename(columns={'class':'TgtType'})

        for i in range(0,df_Labels.shape[0]):
            Frame = df_Labels['frame'][i]
            if str(Frame) not in Ann:
                Ann[str(Frame)] = []
                aspects[str(Frame)] = []
                ranges[str(Frame)] = []
                time_[str(Frame)] = []
            TgtType = df_Labels['TgtType'][i]
            if OptRange:
                TgtRange = df_Labels['range'][i]
            else:
                TgtRange = 0
            TgtAspect=df_Labels['aspect'][i]
            #print(TgtType)
            if (TgtType not in EntityNameDict):
                target_name = 'CLUTTER'
            else:
                target_name = TgtType
                    
            
            if (TgtRange>TargetRangeLimit):
                continue
            curbb = df_Labels['bbox'][i]
            x_min = int(curbb.x)
            y_min = int(curbb.y)
            x_max = int(curbb.x+curbb.w)
            y_max = int(curbb.y+curbb.h)
            #print(curbb)

            if (x_max-x_min<1) or (y_max-y_min<1) or (x_max>xlim_max) or (x_min<2) or (y_max> ylim__max) or (y_max<2):
                continue
            Ann[str(Frame)].append([x_min,y_min,x_max,y_max, target_name])
            ranges[str(Frame)].append(TgtRange)
            time_[str(Frame)].append(get_time(df_Labels['startTime'][i], df_Labels['stopTime'][i]))
            if OptRange:
                if TgtAspect < 45:
                    aspects[str(Frame)].append(1)
                elif TgtAspect >= 45 and TgtAspect < 90 :
                    aspects[str(Frame)].append(2)
                elif TgtAspect >= 90 and TgtAspect < 135:
                    aspects[str(Frame)].append(3)
                elif TgtAspect >= 135 and TgtAspect < 180:
                    aspects[str(Frame)].append(4)
                elif TgtAspect >= 180 and TgtAspect < 225:
                    aspects[str(Frame)].append(5)
                elif TgtAspect >= 225 and TgtAspect < 270:
                    aspects[str(Frame)].append(6)
                elif TgtAspect >= 270 and TgtAspect < 315:
                    aspects[str(Frame)].append(7)
                else:
                    aspects[str(Frame)].append(8)
            else:
                aspects[str(Frame)].append(0)
        for Frame in Ann:
            if (len(Ann[Frame]) == 0) or ((int(Frame)-1)%NSR > 0):
                continue
            AnnList = [os.path.join(DataRoot, FileName)+'.arf', int(Frame)-1, Ann[str(Frame)], aspects[str(Frame)], ranges[str(Frame)], time_[str(Frame)]]
            InputLines.append(AnnList)
    return InputLines
