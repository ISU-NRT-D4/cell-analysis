#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch



# In[2]:


from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])


# In[3]:


root = 'abdu/CSRNet-pytorch/IDCIA'


# In[4]:


#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_A_test]


# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[14]:


import json

with open('test.json', 'r') as f:
  data = json.load(f)


# In[5]:


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)


# In[6]:


model = CSRNet()


# In[7]:


model = model.to(device)


# In[9]:


checkpoint = torch.load('0model_best.pth.tar')


# In[10]:


model.load_state_dict(checkpoint['state_dict'])


# In[34]:


mae = 0
transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
for i in range(len(data)):
    img = Image.open(data[i]).convert('RGB')
    img = transform(img).cuda()
    # img[0,:,:]=img[0,:,:]-92.8207477031
    # img[1,:,:]=img[1,:,:]-95.2757037428
    # img[2,:,:]=img[2,:,:]-104.877445883
    img = img.to(device)
    #img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    gt_file = h5py.File(data[i].replace('.tiff','.h5').replace('images','ground_truth'),'r')
    #print(data[i])
    groundtruth = np.asarray(gt_file['density'])
    output = model(img.unsqueeze(0))
    #print(f"The prediction is: {int(output.detach().cpu().sum().numpy())} and the ground truth is {np.sum(groundtruth)}")
    mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
    print (i,mae)
    print(len(data))
print (f"MAE for Test data: {mae/len(data)}")


# In[ ]:




