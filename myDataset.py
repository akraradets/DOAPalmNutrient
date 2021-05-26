from os import walk
import pandas as pd
from torch._C import Value
from torchvision.io import read_image
from torch.utils.data import Dataset
import math
import numpy as np

class PalmNutriDataset(Dataset):
    def __init__(self, ground_truth, img_dir, sample_set, transform=None, target_transform=None):
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
        if(sample_set not in ['n17']):
            raise ValueError(f"The sample_set '{sample_set}' is not implemented.")
        self.sample_set = sample_set
        self.img_dir = f"{img_dir}/{sample_set}"
        _, _, filenames = next(walk(self.img_dir))
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.filenames[idx]}"
        index = np.argwhere(np.array(self.tree_names) == self.filenames[idx].split('_')[0])[0][0]
        label = self.n_label[index]
        image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample