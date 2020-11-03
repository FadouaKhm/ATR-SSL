# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 18:25:37 2020

@author: khmaissia
"""
import os
import numpy as np
from tkinter import *
from tkinter import ttk, filedialog
import pandas as pd
from sklearn.manifold import TSNE
import tkinter as tk
#import io
from config import config
class ReadFileApp:

    def __init__(self, master):

        self.label = ttk.Label(master, text = "Select files and set parameters", font=('System', 13, 'bold'))



        self.label.grid(row = 0, column =4)
        self.exp_id = ''
        self.txt_SR = ''
        self.tr_SR = 1
        self.ts_SR = 1
        self.vl_SR = 1
        
        
        self.label0 = ttk.Label(master, text = "")
        self.label0.grid(row = 1, column = 4)
        #self.entry1.pack()
        self.train = []
        self.test = []
        self.val = []

        # ttk.Button(master, text = "Training",
        #            command = self.select_train).grid(row = 5, column = 3)

        # self.labelTR = ttk.Label(master, text = "TR  Sampling Rate")
        # self.labelTR.grid(row = 5, column = 4)
        
        # self.entryTR = ttk.Entry(master)
        # self.entryTR.grid(row = 5, column = 5)

        # ttk.Button(master, text = "Save",
        #            command = self.callbackTR).grid(row = 5, column = 6)  

        # ttk.Button(master, text = "Testing",
        #            command = self.select_test).grid(row = 7, column = 3)    

        # self.labelTS = ttk.Label(master, text = "TS  Sampling Rate")
        # self.labelTS.grid(row = 7, column = 4)
        
        # self.entryTS = ttk.Entry(master)
        # self.entryTS.grid(row = 7, column = 5)
        # ttk.Button(master, text = "Save",
        #            command = self.callbackTS).grid(row = 7, column = 6) 
        
        # ttk.Button(master, text = "Validation",
        #            command = self.select_val).grid(row = 9, column = 3)  
        
        # self.labelVL = ttk.Label(master, text = "Val  Sampling Rate")
        # self.labelVL.grid(row = 9, column = 4)
        
        # self.entryVL = ttk.Entry(master)
        # self.entryVL.grid(row = 9, column = 5)
        # ttk.Button(master, text = "Save",
        #            command = self.callbackVL).grid(row = 9, column = 6) 

        # self.label3 = ttk.Label(master, text = "")
        # self.label3.grid(row = 10, column = 4)
        
        self.label2 = ttk.Label(master, text = "Enter experiment ID", font=('System', 12, 'bold'))
        self.label2.grid(row = 12, column = 4)

        
        self.entry1 = ttk.Entry(master)
        self.entry1.grid(row = 13, column = 4 )
        

        ttk.Button(master, text = "Save Experiment ID",
                   command = self.callback).grid(row = 15, column = 4)  
        self.label4 = ttk.Label(master, text = "")
        self.label4.grid(row = 16, column = 4)
        
        ttk.Button(master, text = "Select input txt files",
                   command = self.select_files).grid(row = 17, column = 3)
        self.labeltxt = ttk.Label(master, text = "txt Sampling Rates")
        self.labeltxt.grid(row = 17, column = 4)
        
        self.entrytxt = ttk.Entry(master)
        self.entrytxt.grid(row = 17, column = 5)
        ttk.Button(master, text = "Save",
                   command = self.callbacktxt).grid(row = 17, column = 6) 
        

        # ttk.Button(master, text = "Collection 2 - Select input txt files",
        #            command = self.select_files).grid(row = 18, column = 3)
        # self.labeltxt = ttk.Label(master, text = "txt Sampling Rates")
        # self.labeltxt.grid(row = 18, column = 4)
        
        # self.entrytxt = ttk.Entry(master)
        # self.entrytxt.grid(row = 18, column = 5)
        # ttk.Button(master, text = "Save",
        #            command = self.callbacktxt).grid(row = 18, column = 6) 
        

        # ttk.Button(master, text = "Collection 3 - Select input txt files",
        #            command = self.select_files).grid(row = 19, column = 3)
        # self.labeltxt = ttk.Label(master, text = "txt Sampling Rates")
        # self.labeltxt.grid(row = 19, column = 4)
        
        # self.entrytxt = ttk.Entry(master)
        # self.entrytxt.grid(row = 19, column = 5)
        # ttk.Button(master, text = "Save",
        #            command = self.callbacktxt).grid(row = 19, column = 6) 
        


        # ttk.Button(master, text = "Generate tSNE",
        #            command = self.get_tsne).grid(row = 19, column = 2)  
        
        ttk.Button(master, text = "Generate tSNE - txt",
                   command = self.get_tsne_txt).grid(row = 20, column = 4)  
        
        ttk.Button(master, text = "EXIT",
                   command = master.destroy).grid(row = 21, column = 4)  

    def callbacktxt(self):
        self.txt_SR = str(self.entrytxt.get())
        print("Training sampling rate: " +self.txt_SR)
        print()

    def get_tsne_txt(self):
        
        _tSNE_DIR = './features/'+'/tSNE_Projection'
        if not os.path.exists(_tSNE_DIR ):
            os.mkdir(_tSNE_DIR)
        _FV_DIR = './features/'+'/Feature_Vectors'
            
        print('CONCATENATING FILES ....')
        #print(self.train, self.test, self.val)
        #s = [self.train, self.test, self.val]
        s = self.txt_files
        sets = [k.split('/')[-1] for k in self.txt_files]
        #srates = [int(self.tr_SR), int(self.ts_SR), int(self.vl_SR)]
        if not self.txt_SR:
            srates = [1 for _ in s]
        else:
            srates = [int(sr) for sr in self.txt_SR.split(',')]
        result  = pd.DataFrame()
        set_name = [f.split('.')[0] for f in sets ]
        i = 0
        set_id = []
        for fileslist in s:
            filePaths = []
            with open(fileslist, 'r') as f:
                for line in f:
                    filePaths.append(line.rstrip('\n').split(",")[0])
            
            SR = srates[i]
            #filePaths = os.listdir(os.path.join(_FV_DIR,fileslist.split('__')[0] ))
            for filename in filePaths:
                filename = config["data_config"]["extraction_method"] +'_'+filename.split('/')[-1]+'_Feat.csv'
                #print(_FV_DIR,fileslist,filename )
                df_csv = pd.read_csv(os.path.join(_FV_DIR,fileslist.split('/')[-1].split('__')[0]  ,filename), index_col=None, header=0)
                df_csv = df_csv.loc[df_csv['Frame_ID'].isin(np.unique(df_csv['Frame_ID'])[np.arange(0,len(np.unique(df_csv['Frame_ID'])),SR)])]
                
                #if set_name[i] in ['Train', 'Test', 'Val']:
                df_csv = df_csv.loc[df_csv['Aug_Type'] == 'None']
                
                set_id = set_id + [set_name[i]]*len(df_csv)
                #print(df_csv.shape)
                result = pd.concat([result, df_csv], ignore_index=True)
            i = i+1
        print('Done')
        Meta = result[['GT_YOLO', 'yolo_conf', 'Frame_ID', 'Class_ID', 'Orig_labels','Target Name', 'Aug_IDX', 'Aug_Type', 'Range', 'Time', 'Aspects','Contrast', 'bb_size', 'hw_ratio', 'Collection']]
        Features = result[["F"+str(k) for k in range(1,129)]]
        print('GENERATING tSNE PROJECTON ....')
        Tsn = TSNE(n_components=2).fit_transform(Features)
         
        Meta.insert(0, "tsne_F2", list(Tsn[:,1]), True) 
        Meta.insert(0, "tsne_F1", list(Tsn[:,0]), True) 
        Meta.insert(0, "Set", set_id, True)
        #Meta[["tsne_F1", "tsne_F2"]] = pd.DataFrame(Tsn, index = Meta.index)
        if not self.exp_id:
            pathname = os.path.join(_tSNE_DIR, "tSNE_meta_.csv")
        else:
            pathname = os.path.join(_tSNE_DIR, "tSNE_meta_"+ self.exp_id +".csv")
            
                
        Meta.to_csv(pathname, header=True, index=False)    
        print('tSNE projection complete.')
        print('Output file saved under : ' + pathname)
        print("\n ******* PRESS EXIT ****")

    def select_train(self):
            print()
            print('LOADING TRAINING FILES ....')
            files = filedialog.askopenfilename(multiple=True)     
            self.train = list(files)
            #self.infile = io.TextIOWrapper(self.infile, encoding='utf8', newline='')
            print('Seleted training files:')
            print( [k.split('/')[-1] for k in self.train] )
            print()
            
    def select_files(self):
            print()
            print('LOADING LIST OF FILES ....')
            files = filedialog.askopenfilename(multiple=True)     
            self.txt_files = list(files)
            #self.infile = io.TextIOWrapper(self.infile, encoding='utf8', newline='')
            print('Seleted list of files:')
            print( [k.split('/')[-1] for k in self.txt_files] )
            # print()
            #print(self.txt_files)
            print()

    def callback(self):
        self.exp_id = str(self.entry1.get())
        print("Experiment name: " +self.exp_id)
        print()
        
    def callbackTR(self):
        self.tr_SR = str(self.entryTR.get())
        print("Training sampling rate: " +self.tr_SR)
        print()

    def callbackTS(self):
        self.ts_SR = str(self.entryTS.get())
        print("Testing sampling rate: " +self.ts_SR)
        print()

    def callbackVL(self):
        self.vl_SR = str(self.entryVL.get())
        print("Validation sampling rate: " +self.vl_SR)
        print()

    def select_test(self):
            print('LOADING TESTING FILES ....')
            files = filedialog.askopenfilename(multiple=True)     
            self.test = list(files)
            #self.infile = io.TextIOWrapper(self.infile, encoding='utf8', newline='')
            print('Seleted testing files:')
            print([k.split('/')[-1] for k in self.test])
            print()
            
    def select_val(self):
            print('LOADING VALIDATION FILES ....')
            files = filedialog.askopenfilename(multiple=True)     
            self.val = list(files)
            #self.infile = io.TextIOWrapper(self.infile, encoding='utf8', newline='')
            print('Seleted validation files:')
            print([k.split('/')[-1] for k in self.val])
            print()
            
    def get_tsne(self):
        
        _tSNE_DIR = './features/'+'/tSNE_Projection'
        if not os.path.exists(_tSNE_DIR ):
            os.mkdir(_tSNE_DIR)
            
        print('CONCATENATING FILES ....')
        #print(self.train, self.test, self.val)
        s = [self.train, self.test, self.val]
        srates = [int(self.tr_SR), int(self.ts_SR), int(self.vl_SR)]
        result  = pd.DataFrame()
        set_name = ['Train', 'Test', 'Val']
        i = 0
        set_id = []
        for filePaths in s:

            SR = srates[i]
            
            for filename in filePaths:
                df_csv = pd.read_csv(filename, index_col=None, header=0)
                df_csv = df_csv.loc[df_csv['Frame_ID'].isin(np.unique(df_csv['Frame_ID'])[np.arange(0,len(np.unique(df_csv['Frame_ID'])),SR)])]
                
                if set_name[i] in ['Train', 'Test', 'Val']:
                    df_csv = df_csv.loc[df_csv['Aug_Type'] == 'None']
                
                set_id = set_id + [set_name[i]]*len(df_csv)
                #print(df_csv.shape)
                result = pd.concat([result, df_csv], ignore_index=True)
            i = i+1
        print('Done')
        Meta = result[['GT_YOLO', 'yolo_conf', 'Frame_ID', 'Class_ID', 'Orig_labels','Target Name', 'Aug_IDX', 'Aug_Type', 'Range', 'Time', 'Aspects','Contrast', 'bb_size', 'hw_ratio', 'Collection']]
        Features = result[["F"+str(k) for k in range(1,129)]]
        print('GENERATING tSNE PROJECTON ....')
        Tsn = TSNE(n_components=2).fit_transform(Features)
         
        Meta.insert(0, "tsne_F2", list(Tsn[:,1]), True) 
        Meta.insert(0, "tsne_F1", list(Tsn[:,0]), True) 
        Meta.insert(0, "Set", set_id, True)
        #Meta[["tsne_F1", "tsne_F2"]] = pd.DataFrame(Tsn, index = Meta.index)
        if not self.exp_id:
            pathname = os.path.join(_tSNE_DIR, "tSNE_meta_.csv")
        else:
            pathname = os.path.join(_tSNE_DIR, "tSNE_meta_"+ self.exp_id +".csv")
            
                
        Meta.to_csv(pathname, header=True, index=False)    
        print('tSNE projection complete.')
        print('Output file saved under : ' + pathname)
        print("\n ******* PRESS EXIT ****")
            
        #print(Meta.columns())

def tsne_projection():              
    main = Tk()
    #main.geometry("100x200")
    app = ReadFileApp(main)
    
    main.mainloop()

#if __name__ == "__main__": main()




#def tsne_projection():
#    root = tk.Tk()
#    root.withdraw()
#    root.call('wm', 'attributes', '.', '-topmost', True)
#    files = filedialog.askopenfilename(multiple=True) 
#    #%gui tk
#    var = root.tk.splitlist(files)
#    filePaths = []
#    for f in var:
#        filePaths.append(f)
#    filePaths
#    
#    ## 2- LOAD CSVs and CONCAT PROJECTIONS
#    
#    result = pd.DataFrame()
#    tsne_filename = ''
#    for filename in filePaths:
#        df_csv = pd.read_csv(filename, index_col=None, header=0)
#        #print(df_csv.shape)
#        result = pd.concat([result, df_csv], ignore_index=True)
#        tsne_filename = tsne_filename + "_"+ filename.split("/")[-1].split(".")[0][:-5]
#        
#    Meta = result[["GT_Yolo", "Label", "Yolo confidence", "Frame_ID", "Target_name", "Time", "Range", "Width X Height", "Aug_IDX", "Aug_Type", "Collection"]]
#    Features = result[["F"+str(k) for k in range(1,129)]]
#    Tsn = TSNE(n_components=2).fit_transform(Features)
#    Meta[["tsne_F1", "tsne_F2"]] = pd.DataFrame(Tsn, index = Meta.index)
#    
#    Meta.to_csv(os.path.join("./features/"+ "tsne_meta_"+tsne_filename+".csv"), header=True, index=False)
#
#import tkinter
#from tkinter import messagebox
#
#top = tkinter.Tk()
#def hello():
#   messagebox.showinfo("Say Hello", "Hello World")
#
#B1 = tkinter.Button(top, text = "Select training files", command = tsne_projection)
#B1.pack()
#
#top.mainloop()