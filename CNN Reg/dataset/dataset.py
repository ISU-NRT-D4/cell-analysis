from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import random
import torchvision.transforms as transforms

import pandas as pd
import torch
#Binarize mask
def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[mask == 0.0] = 0.0
    mask[(mask > 0.0) ] = 1.0
    return mask


class IDCIA(Dataset):
    def __init__(self, images_filenames, images_directory, label_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.label_directory = label_directory
        self.transform = transform
        
    def __len__(self): 
        return len(self.images_filenames)


    def __getitem__(self, idx):
        #Get a record of filename
        image_filename = self.images_filenames[idx]

        #Read Image and respective mask as numpy arrays
        image=np.array(Image.open(os.path.join(self.images_directory, image_filename)).convert("RGB"),dtype=np.uint8)
        label=pd.read_csv(os.path.join(self.label_directory,image_filename[:-5]+".csv")).shape[0]

     
       
        #Apply transforms on image
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        image=transforms.ToTensor()(image)
        label=torch.tensor(label,dtype=torch.float32).unsqueeze(-1)
        
        return image,label
