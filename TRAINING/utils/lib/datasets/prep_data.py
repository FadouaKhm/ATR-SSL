import numpy as np
import os

class data:
    #def __init__(self, root, split="l_train"):
    #   self.dataset = np.load(os.path.join(root, "data", split+".npy"), allow_pickle=True).item()

    def __init__(self, X):
        self.dataset = X
        
        
    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        #image = (image - 0.5)/0.5
        #image = 1-image
        return image, label

    def __len__(self):
        return len(self.dataset["images"])