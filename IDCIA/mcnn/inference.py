import os
import torch
import numpy as np

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils
import os
import pandas
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import random
import cv2
import pandas as pd
import matplotlib.pyplot as plt

print(torch.cuda.is_available())

test_samples = ["220909_GFP-AHPC_D_MAP2ab_F7_Cy3_ND2_20x.tiff",
                "220912_GFP-AHPC_B_RIP_F6_Cy3_ND2_20x.tiff",
                "220912_GFP-AHPC_A_Nestin_F3_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_D_Ki67_F2_DAPI_ND1_20x.tiff",
                "220912_GFP-AHPC_B_Nestin_F1_Cy3_ND2_20x.tiff",
                "220909_GFP-AHPC_D_TuJ1_F5_Cy3_ND2_20x.tiff",
                "220909_GFP-AHPC_D_MAP2ab_F7_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_D_Ki67_F9_Cy3_ND8_20x.tiff",
                "220909_GFP-AHPC_D_Nestin_F9_DAPI_ND1_20x.tiff",
                "220912_GFP-AHPC_A_Nestin_F8_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_D_MAP2ab_F9_Cy3_ND2_20x.tiff",
                "220912_GFP-AHPC_B_RIP_F3_Cy3_ND2_20x.tiff",
                "220912_GFP-AHPC_A_TuJ1_F8_Cy3_ND2_20x.tiff",
                "220912_GFP-AHPC_B_RIP_F5_Cy3_ND2_20x.tiff",
                "220909_GFP-AHPC_D_GFAP_F7_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_D_TuJ1_F6_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_C_Ki67_F7_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_D_Ki67_F10_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_C_GFAP_F4_Cy3_ND2_20x.tiff",
                "220912_GFP-AHPC_C_TuJ1_F1.2_Cy3_ND1_20x.tiff",
                "220909_GFP-AHPC_B_GFAP_F8_DAPI_ND1_20x.tiff",
                "220912_GFP-AHPC_A_Nestin_F2_Cy3_ND2_20x.tiff",
                "220909_GFP-AHPC_D_Nestin_F8_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_A_Ki67_F9_CY3_ND8_20x.tiff",
                "220909_GFP-AHPC_D_Nestin_F4_DAPI_ND1_20x.tiff",
                "220912_GFP-AHPC_B_TuJ1_F4_Cy3_ND2_20x.tiff",
                "220909_GFP-AHPC_D_RIP_F3_Cy3_ND2_20x.tiff",
                "220912_GFP-AHPC_C_TuJ1_F8_DAPI_ND1_20x.tiff",
                "220912_GFP-AHPC_C_Map2AB_F5_DAPI_ND1_20x.tiff",
                "220912_GFP-AHPC_A_Map2AB_F4_Cy3_ND2_20x.tiff",
                "220912_GFP-AHPC_B_Map2AB_F2_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_D_MAP2ab_F2_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_D_TuJ1_F5_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_C_Ki67_F9_Cy3_ND8_20x.tiff",
                "220912_GFP-AHPC_B_TuJ1_F8_Cy3_ND2_20x.tiff",
                "220912_GFP-AHPC_B_Map2AB_F9_Cy3_ND2_20x.tiff",
                "220909_GFP-AHPC_D_MAP2ab_F1_Cy3_ND2_20x.tiff",
                "220909_GFP-AHPC_A_GFAP_F6_Cy3_ND2_20x.tiff",
                "220909_GFP-AHPC_D_Nestin_F8_Cy3_ND2_20x.tiff",
                "220909_GFP-AHPC_C_GFAP_F6_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_B_GFAP_F8_Cy3_ND2_20x.tiff",
                "220909_GFP-AHPC_B_GFAP_F3_Cy3_ND2_20x.tiff",
                "220909_GFP-AHPC_C_GFAP_F9_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_C_Nestin_F6_Cy3_ND2_20x.tiff",
                "220909_GFP-AHPC_D_GFAP_F8_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_D_RIP_F4_Cy3_ND2_20x.tiff",
                "220912_GFP-AHPC_A_TuJ1_F6_DAPI_ND1_20x.tiff",
                "220912_GFP-AHPC_A_TuJ1_F7_DAPI_ND1_20x.tiff",
                "220912_GFP-AHPC_A_Nestin_F6_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_B_Ki67_F5_DAPI_ND1_20x.tiff",
                "220909_GFP-AHPC_B_Ki67_F5_Cy3_ND8_20x.tiff",
                "220909_GFP-AHPC_B_Ki67_F3_Cy3_ND8_20x.tiff",
                "220909_GFP-AHPC_D_GFAP_F2_Cy3_ND2_20x.tiff"]
test_samples = ["220909_GFP-AHPC_D_MAP2ab_F7_Cy3_ND2_20x.tiff",
                                 "220912_GFP-AHPC_B_RIP_F6_Cy3_ND2_20x.tiff",
                                 "220912_GFP-AHPC_A_Nestin_F3_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_D_Ki67_F2_DAPI_ND1_20x.tiff",
                                 "220912_GFP-AHPC_B_Nestin_F1_Cy3_ND2_20x.tiff",
                                 "220909_GFP-AHPC_D_TuJ1_F5_Cy3_ND2_20x.tiff",
                                 "220909_GFP-AHPC_D_MAP2ab_F7_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_D_Ki67_F9_Cy3_ND8_20x.tiff",
                                 "220909_GFP-AHPC_D_Nestin_F9_DAPI_ND1_20x.tiff",
                                 "220912_GFP-AHPC_A_Nestin_F8_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_D_MAP2ab_F9_Cy3_ND2_20x.tiff",
                                 "220912_GFP-AHPC_B_RIP_F3_Cy3_ND2_20x.tiff",
                                 "220912_GFP-AHPC_A_TuJ1_F8_Cy3_ND2_20x.tiff",
                                 "220912_GFP-AHPC_B_RIP_F5_Cy3_ND2_20x.tiff",
                                 "220909_GFP-AHPC_D_GFAP_F7_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_D_TuJ1_F6_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_C_Ki67_F7_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_D_Ki67_F10_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_C_GFAP_F4_Cy3_ND2_20x.tiff",
                                 "220912_GFP-AHPC_C_TuJ1_F1.2_Cy3_ND1_20x.tiff",
                                 "220909_GFP-AHPC_B_GFAP_F8_DAPI_ND1_20x.tiff",
                                 "220912_GFP-AHPC_A_Nestin_F2_Cy3_ND2_20x.tiff",
                                 "220909_GFP-AHPC_D_Nestin_F8_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_A_Ki67_F9_CY3_ND8_20x.tiff",
                                 "220909_GFP-AHPC_D_Nestin_F4_DAPI_ND1_20x.tiff",
                                 "220912_GFP-AHPC_B_TuJ1_F4_Cy3_ND2_20x.tiff",
                                 "220909_GFP-AHPC_D_RIP_F3_Cy3_ND2_20x.tiff",
                                 "220912_GFP-AHPC_C_TuJ1_F8_DAPI_ND1_20x.tiff",
                                 "220912_GFP-AHPC_C_Map2AB_F5_DAPI_ND1_20x.tiff",
                                 "220912_GFP-AHPC_A_Map2AB_F4_Cy3_ND2_20x.tiff",
                                 "220912_GFP-AHPC_B_Map2AB_F2_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_D_MAP2ab_F2_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_D_TuJ1_F5_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_C_Ki67_F9_Cy3_ND8_20x.tiff",
                                 "220912_GFP-AHPC_B_TuJ1_F8_Cy3_ND2_20x.tiff",
                                 "220912_GFP-AHPC_B_Map2AB_F9_Cy3_ND2_20x.tiff",
                                 "220909_GFP-AHPC_D_MAP2ab_F1_Cy3_ND2_20x.tiff",
                                 "220909_GFP-AHPC_A_GFAP_F6_Cy3_ND2_20x.tiff",
                                 "220909_GFP-AHPC_D_Nestin_F8_Cy3_ND2_20x.tiff",
                                 "220909_GFP-AHPC_C_GFAP_F6_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_B_GFAP_F8_Cy3_ND2_20x.tiff",
                                 "220909_GFP-AHPC_B_GFAP_F3_Cy3_ND2_20x.tiff",
                                 "220909_GFP-AHPC_C_GFAP_F9_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_C_Nestin_F6_Cy3_ND2_20x.tiff",
                                 "220909_GFP-AHPC_D_GFAP_F8_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_D_RIP_F4_Cy3_ND2_20x.tiff",
                                 "220912_GFP-AHPC_A_TuJ1_F6_DAPI_ND1_20x.tiff",
                                 "220912_GFP-AHPC_A_TuJ1_F7_DAPI_ND1_20x.tiff",
                                 "220912_GFP-AHPC_A_Nestin_F6_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_B_Ki67_F5_DAPI_ND1_20x.tiff",
                                 "220909_GFP-AHPC_B_Ki67_F5_Cy3_ND8_20x.tiff",
                                 "220909_GFP-AHPC_B_Ki67_F3_Cy3_ND8_20x.tiff",
                                 "220909_GFP-AHPC_D_GFAP_F2_Cy3_ND2_20x.tiff"]
model_path = 'src/saved_models/best_model.h5'

net = CrowdCounter()
network.load_net(model_path, net)
net.eval()
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
])
root = os.path.join("data", "IDCIA", "images")

'''
for i in range(5):
    ch = random.choice(test_samples)
    print(f"Counting from {ch}")
    img = cv2.imread(os.path.join(root, ch), 0)
    ht = img.shape[0]
    wd = img.shape[1]
    ht_1 = (ht / 4) * 4
    wd_1 = (wd / 4) * 4

    img = cv2.resize(img, (int(wd_1), int(ht_1)))
    print(img.shape)

    img = img.reshape((1, 1, img.shape[0], img.shape[1]))

    net.cuda()
    pred = net(img).data.cpu().numpy()

    print(f"prediction: {np.sum(pred)}")
    df = pd.read_csv('../cell_counting/data/IDCIA/ground_truth/' + ch.split("/")[-1].replace('tiff', 'csv'))
    print(f"Actual count: {df.shape[0]}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].imshow(img.squeeze(0).squeeze(0), cmap="gray")
    ax[0].text(0.5, -0.1, f"Actual count: {df.shape[0]}", size=12, ha="center",
               transform=ax[0].transAxes)
    ax[1].imshow(pred.squeeze(0).squeeze(0))
    ax[1].text(0.5, -0.1, f"Predicted count: {round(np.sum(pred))}", size=12, ha="center",
               transform=ax[1].transAxes)
    plt.show()
'''
report={}
for i in test_samples:
    img = cv2.imread(os.path.join(root, i), 0)
    ht = img.shape[0]
    wd = img.shape[1]
    ht_1 = (ht / 4) * 4
    wd_1 = (wd / 4) * 4

    img = cv2.resize(img, (int(wd_1), int(ht_1)))


    img = img.reshape((1, 1, img.shape[0], img.shape[1]))

    net.cuda()
    pred = net(img).data.cpu().numpy()
    #pred = net(img).cpu().data.numpy()

    print(f"prediction: {np.sum(pred)}")
    df = pd.read_csv('../cell_counting/data/IDCIA/ground_truth/' + i.split("/")[-1].replace('tiff', 'csv'))
    print(f"Actual count: {df.shape[0]}")
    report[i] = (df.shape[0], np.sum(pred))
pd.DataFrame(report).transpose().to_csv("report_mcnn.csv", index=True, header=True)

