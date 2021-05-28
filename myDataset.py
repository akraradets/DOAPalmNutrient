from os import walk
import pandas as pd
from torch._C import Value
# from torchvision.io import read_image
from torch.utils.data import Dataset
import math
import numpy as np
from PIL import Image
import pickle

class PalmNutriDataset(Dataset):
    def __init__(self, ground_truth, img_dir, sample_set, filter = True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        
        df = pd.read_csv(ground_truth)
        self.tree_names = np.array(df['name'])
        self.n_label = df['N']
        self.k_label = df['K']
        self.n_range = [0,2,2.4,3,math.inf]
        self.k_range = [0,0.75,0.90,1.2,math.inf]
        
        if(sample_set not in ['n17','n33','k17','k33']):
            raise ValueError(f"the sample_set '{sample_set}' is not support. Only {['n17','n33','k17','k33']} is valid.")
        # if(sample_set not in ['n17','n33']):
        #     raise ValueError(f"The sample_set '{sample_set}' is not implemented.")
        self.sample_set = sample_set
        self.img_dir = f"{img_dir}/{sample_set}"
        _, _, filenames = next(walk(self.img_dir))
        filt_list = []
        if(filter):
            if(sample_set == 'k17'):
                with open('dataset/k17_filter.pickle', 'rb') as handle:
                    filt_list = pickle.load(handle)
            elif(sample_set == 'k33'):
                with open('dataset/k33_filter.pickle', 'rb') as handle:
                    filt_list = pickle.load(handle)
        for name in filenames:
            if(int(name.split('_')[0][1:3]) in filt_list):
                filenames.remove(name)

        self.filenames = filenames            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.filenames[idx]}"
        try:
            index = np.argwhere(np.array(self.tree_names) == self.filenames[idx].split('_')[0])[0][0]
        except:
            print(self.filenames[idx])
            raise ValueError
        if(self.sample_set in ['n17','n33']):
            label = self.n_label[index]
        else:
            label = self.k_label[index]
        image = Image.open(img_path)
        # label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        import torch
        sample = (image,torch.tensor(label,dtype=torch.float32))
        return sample