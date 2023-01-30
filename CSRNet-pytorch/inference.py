import sys
import os

import pandas as pd
import matplotlib.pyplot as plt

import warnings

from model import CSRNet

from utils import save_checkpoint
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time
from PIL import Image
import cv2
import csv
from tqdm import tqdm

def main():
    print("Counting Cells...")
    checkpoint = torch.load("0model_best.pth.tar")
    epoch = checkpoint['epoch']
    model = CSRNet()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
    ])

    test_samples = [
                                 "IDCIA/images/220909_GFP-AHPC_D_MAP2ab_F7_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_B_RIP_F6_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_A_Nestin_F3_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_Ki67_F2_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_B_Nestin_F1_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_TuJ1_F5_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_MAP2ab_F7_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_Ki67_F9_Cy3_ND8_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_Nestin_F9_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_A_Nestin_F8_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_MAP2ab_F9_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_B_RIP_F3_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_A_TuJ1_F8_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_B_RIP_F5_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_GFAP_F7_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_TuJ1_F6_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_C_Ki67_F7_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_Ki67_F10_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_C_GFAP_F4_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_C_TuJ1_F1.2_Cy3_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_B_GFAP_F8_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_A_Nestin_F2_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_Nestin_F8_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_A_Ki67_F9_CY3_ND8_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_Nestin_F4_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_B_TuJ1_F4_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_RIP_F3_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_C_TuJ1_F8_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_C_Map2AB_F5_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_A_Map2AB_F4_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_B_Map2AB_F2_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_MAP2ab_F2_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_TuJ1_F5_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_C_Ki67_F9_Cy3_ND8_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_B_TuJ1_F8_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_B_Map2AB_F9_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_MAP2ab_F1_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_A_GFAP_F6_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_Nestin_F8_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_C_GFAP_F6_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_B_GFAP_F8_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_B_GFAP_F3_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_C_GFAP_F9_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_C_Nestin_F6_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_GFAP_F8_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_RIP_F4_Cy3_ND2_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_A_TuJ1_F6_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_A_TuJ1_F7_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220912_GFP-AHPC_A_Nestin_F6_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_B_Ki67_F5_DAPI_ND1_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_B_Ki67_F5_Cy3_ND8_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_B_Ki67_F3_Cy3_ND8_20x.tiff",
                                 "IDCIA/images/220909_GFP-AHPC_D_GFAP_F2_Cy3_ND2_20x.tiff"]

    # for i in range(5):
    #     ch = random.choice(test_samples)
    #     img = Image.open(ch).convert("RGB")
    #     model.eval()
    #     pred = model(transform(img))
    #     print(f"Prediction shape: {pred.shape}")
    #     df = pd.read_csv('../cell_counting/data/IDCIA/ground_truth/' + ch.split("/")[-1].replace('tiff', 'csv'))
    #
    #     fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    #     print(ch.split("/")[-1].replace('tiff', 'csv'))
    #     ax[0].axis('off')
    #     ax[1].axis('off')
    #     ax[0].imshow(img)
    #     ax[0].text(0.5, -0.1, f"Actual count: {df.shape[0]}", size=12, ha="center",
    #                transform=ax[0].transAxes)
    #     ax[1].imshow(pred.squeeze().detach())
    #     ax[1].text(0.5, -0.1, f"Predicted count: {round(pred.sum().item())}", size=12, ha="center",
    #                transform=ax[1].transAxes)
    #     plt.show()
    #
    #
    #     print(f"Actual count: {df.shape[0]}")
    #     print(f"Prediction count: {round(pred.sum().item())}")
    report = {}
    for i in tqdm(test_samples):
        img = Image.open(i).convert("RGB")
        model.eval()
        pred = model(transform(img))

        df = pd.read_csv('../cell_counting/data/IDCIA/ground_truth/' + i.split("/")[-1].replace('tiff', 'csv'))



        #print(i.split("/")[-1].replace('tiff', 'csv'))

        #print(f"Actual count: {df.shape[0]}")

        #print(f"Prediction count: {round(pred.sum().item())}")
        report[i]=(df.shape[0],round(pred.sum().item()))
    pd.DataFrame(report).transpose().to_csv("report.csv",index=True,header=True)


if __name__ == '__main__':
    main()
