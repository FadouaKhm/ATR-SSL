# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:37:53 2020

@author: Khmaissia
"""


from tkinter.ttk import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from config import config

import os
from tkinter import *
from tkinter import ttk, filedialog
import pandas as pd
from sklearn.manifold import TSNE
import tkinter as tk
import numpy as np
from pandas.api.types import CategoricalDtype
#import io
def gen_unique_v(data, selected_features , continuous_vars):
    if selected_features[0] in continuous_vars:
        ref_name = selected_features[0]
        ref_thresh = np.percentile(data[selected_features[0]], [25, 50, 75])
        [mn, mx] = [np.min(data[selected_features[0]]), np.max(data[selected_features[0]])]
        z = np.digitize(data[selected_features[0]], np.percentile(data[selected_features[0]], [25, 50, 75]))
        uniques = ['{:.2f}'.format(mn) + ' <= ' + ref_name+' < ' +'{:.2f}'.format(ref_thresh[0]), '{:.2f}'.format(ref_thresh[0])+' <= ' + ref_name+' < ' +'{:.2f}'.format(ref_thresh[1]), '{:.2f}'.format(ref_thresh[1])+' <= ' + ref_name+' < ' +'{:.2f}'.format(ref_thresh[2]) ,'{:.2f}'.format(ref_thresh[2]) +' <= ' + ref_name+' <= ' +'{:.2f}'.format(mx)]
    
#    elif selected_features[0] == 'Range':
#        z1 = np.digitize(data[selected_features[0]], np.array(config["Ranges"])+100)
#        z1 =np.array( [config["Ranges"][i] for i in z1])
#        cat_dtype = pd.api.types.CategoricalDtype(categories = pd.unique(z1), ordered=False)
#        z = np.array(pd.Series(z1).astype(cat_dtype).cat.codes)
#        
#        #uniques = np.unique(np.array(data[selected_features[0]]))
#        uniques = pd.unique(data[selected_features[0]])
    else:
        uniques = pd.unique(data[selected_features[0]])
        cat_dtype = pd.api.types.CategoricalDtype(categories = uniques, ordered=False)
        
        z = np.array(data[selected_features[0]].astype(cat_dtype).cat.codes)  
        if selected_features[0] == 'Class_ID':
           lbls = ['Civilian', 'Military']
           uniques = [lbls[i] for i in uniques]
           
#        print(uniques,"\n")
#        print(z[:100],"\n")
#        print(data[selected_features[0]][:100],"\n")
    
    return z, uniques 
class Checkbar(Frame):
   def __init__(self, parent=None, picks=[], side=LEFT, anchor=W):
      Frame.__init__(self, parent)
      
      self.vars = []
      for pick in picks:
         var = IntVar()
         chk = Checkbutton(self, text=pick, variable=var, offvalue=0)
         chk.deselect()
         chk.pack(side=side, anchor=anchor, expand=YES)
         self.vars.append(var)
   def state(self):
      return map((lambda var: var.get()), self.vars)
  
   def clear(self):
      #return map((lambda var: var.set(0)), self.vars)
      for var in self.vars:
          var.set(0)
      #return [var.pack_forget() for var in self.vars]
  
    
class LoadFileApp:

    def __init__(self, master):

        self.label = ttk.Label(master, text = "Select tSNE files", font=('System', 12, 'bold'))
        self.label.grid(row = 0, column = 3, columnspan = 3)
        self.exp_id = ''
        self.df_csv = pd.DataFrame()

        #self.entry1.pack()
        self.train = []
        self.test = []
        self.val = []

        ttk.Button(master, text = "Select tSNE Feature Vector",
                   command = self.select_tsne).grid(row = 5, column = 3) 
        ttk.Button(master, text = "Load",
                   command = self.load_tsne).grid(row = 6, column = 3)         
        ttk.Button(master, text = "EXIT",
                   command = master.destroy).grid(row = 14, column = 3)  

   
    def select_tsne(self):
            print()
            print('LOADING tSNE FEATURE VECTOR ....')
            files = filedialog.askopenfilename(multiple=False)     
            self.tsne_f = files
            #self.infile = io.TextIOWrapper(self.infile, encoding='utf8', newline='')
            print('Seleted feature file:')
            print( [self.tsne_f] )
            print()

            
    def load_tsne(self):
        
        _tSNE_DIR = './features/'+'/tSNE_Projection'
        if not os.path.exists(_tSNE_DIR ):
            os.mkdir(_tSNE_DIR)
            
        print('Features: ')
        self.df_csv = pd.read_csv(self.tsne_f, index_col=None, header=0)
        print(self.df_csv.columns)
        print('Done \n ******* PRESS EXIT *****')
        return (self.df_csv)

        
        
    def get_file(self):
        return (self.df_csv)

          
class gen_plots:   
    def __init__(self):
        self.t = []
    def vizualize(self):   
    #    global data    
    #    global selected_features    
        main = Tk()
        #main.geometry("200x300")
        #main.resizable(1, 1)
        app = LoadFileApp(main)
    
        main.mainloop()
        
        self.data = app.df_csv
        
        frame = Tk()

        root = Frame(frame)
        root.pack()
        
        
        self.features = ['Set', 'GT_YOLO', 'Class_ID', 'Target Name', 'Aug_Type', 'Range', 'Time', 'Aspects', 'Contrast', 'bb_size', 'hw_ratio', 'Collection']
        self.lng = Checkbar(root, self.features)
        self.symbols =  ['Set', 'GT_YOLO', 'Class_ID', 'Target Name', 'Aug_Type', 'Range', 'Time', 'Aspects', 'Contrast', 'bb_size', 'hw_ratio', 'Collection']#['Set', 'GT_YOLO', 'Class_ID', 'Target Name', 'Aug_Type', 'Collection'] #['Set' , 'Class_ID', 'Target Name','Aug_Type'  ]
        self.label = Label(root, text = "Color-coded features:", font=('System', 12, 'bold'))
        self.label.pack(fill=X)
        self.tgl = Checkbar(root, self.symbols)
        self.lng.pack(side=TOP,  fill=X)

        self.label2 = Label(root, text = "Symbol-coded features:", font=('System', 12, 'bold'))
        self.label2.pack(fill=X)
        self.tgl.pack(side=LEFT)
        self.lng.config(relief=GROOVE, bd=2)
        
        #global_vars.selected_features = list(np.array(features)[np.nonzero(list(lng.state()))[0]])
    
    
        def allstates(self): 
           print(list(self.lng.state()))
           
        def reset(): 
            return self.lng.clear(), self.tgl.clear()
            
            

        
        def plot_s(): 
            from matplotlib.lines import Line2D
            from matplotlib import cm
            print ("\n ****** PROCESSING PLOTS .....")
            if list(np.array(self.features)[np.nonzero(list(self.lng.state()))[-1]]):
                self.selected_features = [list(np.array(self.features)[np.nonzero(list(self.lng.state()))[-1]])[-1]]
            else:
                self.selected_features = ['GT_YOLO']

            if list(np.array(self.symbols)[np.nonzero(list(self.tgl.state()))[-1]]):
                self.selected_symbols = [list(np.array(self.symbols)[np.nonzero(list(self.tgl.state()))[-1]])[-1]]
            else:
                self.selected_symbols = ['GT_YOLO']

#            if self.selected_features in ['Contrast', 'bb_size', 'hw_ratio']:
#                ref_name = self.selected_features
#                ref_thresh = np.percentile(self.data[self.selected_features], [25, 50, 75])
#                [mn, mx] = [np.min(self.data[self.selected_features]), np.max(self.data[self.selected_features])]
#                z = np.digitize(self.data[self.selected_features], np.percentile(self.data[self.selected_features], [25, 50, 75]))
#                uniques = ['{:.2f}'.format(mn) + ' <= ' + ref_name+' < ' +'{:.2f}'.format(ref_thresh[0]), '{:.2f}'.format(ref_thresh[0])+' <= ' + ref_name+' < ' +'{:.2f}'.format(ref_thresh[1]), '{:.2f}'.format(ref_thresh[1])+' <= ' + ref_name+' < ' +'{:.2f}'.format(ref_thresh[2]) ,'{:.2f}'.format(ref_thresh[2]) +' <= ' + ref_name+' <= ' +'{:.2f}'.format(mx)]
#            else:
#                uniques = np.unique(np.array(self.data[self.selected_features]))
#                z = np.array(self.data[self.selected_features].astype('category').cat.codes)
            feat_col, uniques_feat = gen_unique_v(self.data, self.selected_features , ['Contrast', 'bb_size', 'hw_ratio'])
            feat_sym, uniques_syms = gen_unique_v(self.data, self.selected_symbols , ['Contrast', 'bb_size', 'hw_ratio'])
            
            #uniques = np.unique(np.array(self.data[self.selected_features[-1]]))
            cmaps = cm.get_cmap('Paired', len(uniques_feat))
    
#            legend_elements = []
#            
#            for i in range(len(uniques)):
#                legend_elements.append(Line2D([0], [0], marker='o', color=cmaps(i), label= uniques[i], markerfacecolor=cmaps(i) , markersize=5))              
#
            rootp= Tk()
            figure = Figure(figsize=(12,8), dpi= 100)
            plot = figure.add_subplot(1, 1, 1)
            print("Selected features: " , self.selected_features )
            print("Filtered by: " , self.selected_symbols )
            sizes = config["marker_sizes"] #'o.+v38p*hX|_DP>sx'
            markers_list = config["markers_list" ]#[ 20, 10, 200, 30, 35, 40, 45]
#            #print(self.data)
            temp = 'none'
            if self.selected_features[0] in ['Contrast', 'bb_size', 'hw_ratio']:
                temp = self.selected_features[0]
            if self.selected_symbols[0] in ['Contrast', 'bb_size', 'hw_ratio']:
                temp = self.selected_symbols[0]            
#           
#            #print(data)
#            #z = np.array(self.data[self.selected_features[-1]].astype('category').cat.codes)
#            plot.scatter(self.data['tsne_F1'], self.data['tsne_F2'], s = 5,  c = z,cmap = cmaps)
           # groups = self.data.groupby(self.selected_features[0])
            legend_elements = [] 
            #feat_sym = self.data[self.selected_symbols[0]]
            j = 0

            for k in range(len(uniques_syms)):
                syms1 = uniques_syms[k]
                
                  
                for i in range(len(uniques_feat)):
                    
                    syms = k
                    #print(feat_col[:10], uniques_feat)
                    data_i = self.data.loc[feat_col == i]
                    feat_col_i = feat_col[feat_col  == i]
                    feat_sym_i = feat_sym[feat_col == i]
                    if list(feat_sym_i == syms).count(True):
                        plot.scatter(data_i['tsne_F1'][feat_sym_i == syms], data_i['tsne_F2'][feat_sym_i == syms], s = sizes[k],    marker=markers_list[k], color = cmaps(i)) #c = feat_col_i[feat_sym_i == syms] ,
                        legend_elements.append(Line2D([0], [0], marker=markers_list[k], color=cmaps(i), label= str(self.selected_features[0]) + ' = ' + str(uniques_feat[i]) + ', ' +str(self.selected_symbols[0]) + '= ' + str(syms1), markerfacecolor=cmaps(i) , markersize= 10))

            
#                print(selected_features )
#                print(selected_symbols)
                #print(data)
                
                #plot.scatter(self.data['tsne_F1'][feat_sym == syms], self.data['tsne_F2'][feat_sym == syms], s = sizes[k],  c = feat_col[feat_sym == syms] ,  marker=markers_list[k], cmap = cmaps)
                #plot.scatter(data['tsne_F1'], data['tsne_F2'], c = np.array(data[selected_features[0]]),cmap = cmaps)
                j = j+1
            #plot.scatter(data['tsne_F1'], data['tsne_F2'], c = np.array(data[selected_features[0]]),cmap = cmaps)
            plot.legend(handles=legend_elements, bbox_to_anchor=[1, 1])
            plot.set_xlabel('Feature 1 - t-SNE embedding')
            plot.set_ylabel('Feature 2 - t-SNE embedding')
            plot.set_title('tSNE - Selected features : ' + self.selected_features[0]+ " and " + self.selected_symbols[0])
            
            canvas = FigureCanvasTkAgg(figure, rootp)
            canvas.get_tk_widget().grid(row=0, column=2)  
            
            toolbarFrame = Frame(master=rootp)
            toolbarFrame.grid(row=1,column=2)
            toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
            #toolbar.update()
            #canvas.get_tk_widget().grid(row=0, column=2)  

                
            if temp in ['Contrast', 'bb_size', 'hw_ratio']:
                roothist= Tk()
                figure = Figure(figsize=(12,8), dpi= 100)
                plot = figure.add_subplot(1, 1, 1)   
                plot.hist(self.data[temp])
                plot.grid()
                #plot.legend(handles=legend_elements, bbox_to_anchor=[1, 1])
                plot.set_ylabel('Frequencies')
                plot.set_xlabel('Values')
                plot.set_title('Distribution of : ' + temp)
                
                canvas = FigureCanvasTkAgg(figure, roothist)
                canvas.get_tk_widget().grid(row=0, column=2)  
                
                toolbarFrame = Frame(master=roothist)
                toolbarFrame.grid(row=1,column=2)
                toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)


           #print(list(lng.state()))

        bottomframe = Frame(frame)
        bottomframe.pack( side = BOTTOM )

           
        Button(bottomframe, text='EXIT', command=frame.destroy, font=('System', 12, 'bold')).pack(side=BOTTOM)
        
        Button(bottomframe, text='Reset', command=reset, font=('System', 12, 'bold')).pack(side=BOTTOM)
        Button(bottomframe, text='Plot', command=plot_s, font=('System', 12, 'bold')).pack(side=BOTTOM)
        root.mainloop()
    
        #print( "selected features" , np.array(features)[np.nonzero(list(lng.state()))[0]])


if __name__ == "__main__": main()