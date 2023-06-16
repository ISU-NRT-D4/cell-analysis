import os
import torch
import numpy as np
import sys

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model
import math

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)



method = 'mcnn'
dataset_name = 'shtechA'
output_dir = './saved_models/'

train_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train'
train_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
val_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val'
val_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'


train_images_filenames = ["220909_GFP-AHPC_D_MAP2ab_F9_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_A_Nestin_F10_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_C_RIP_F2_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_A_Ki67_F1_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_Nestin_F2_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_B_GFAP_F6_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_Nestin_F2_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_B_Ki67_F3_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_Nestin_F9_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_D_Ki67_F9_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_A_Map2AB_F6_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_B_TuJ1_F5_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_A_Map2AB_F2_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_Nestin_F7_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_C_GFAP_F3_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_Nestin_F3_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_GFAP_F3_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_B_GFAP_F10_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_B_Ki67_F6_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_A_Ki67_F1_CY3_ND8_20x.tiff",
                                  "220909_GFP-AHPC_C_Nestin_F5_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_C_Map2AB_F2_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_A_RIP_F10_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_C_TuJ1_F9_Cy3_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_Nestin_F3_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_Nestin_F1_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_C_TuJ1_F1.1_Cy3_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_Ki67_F5_Cy3_ND8_20x.tiff",
                                  "220909_GFP-AHPC_D_MAP2ab_F5_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_GFAP_F3_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_B_Ki67_F9_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_RIP_F1_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_B_GFAP_F4_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_GFAP_F5_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_B_TuJ1_F4_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_TuJ1_F9_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_Ki67_F2_Cy3_ND8_20x.tiff",
                                  "220912_GFP-AHPC_A_Map2AB_F9_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_GFAP_F7_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_Ki67_F3_Cy3_ND8_20x.tiff",
                                  "220909_GFP-AHPC_A_Ki67_F4_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_GFAP_F7_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_B_RIP_F4_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_C_TuJ1_F10_Cy3_ND1_20x.tiff",
                                  "220912_GFP-AHPC_C_TuJ1_F8_Cy3_ND1_20x.tiff",
                                  "220912_GFP-AHPC_A_TuJ1_F8_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_Ki67_F8_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_A_TuJ1_F5_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_Nestin_F2_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_C_RIP_F6_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_B_RIP_F9_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_A_TuJ1_F9_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_A_Ki67_F3_CY3_ND8_20x.tiff",
                                  "220909_GFP-AHPC_C_Ki67_F7_Cy3_ND8_20x.tiff",
                                  "220909_GFP-AHPC_A_Ki67_F4_CY3_ND8_20x.tiff",
                                  "220909_GFP-AHPC_A_Ki67_F7_CY3_ND8_20x.tiff",
                                  "220912_GFP-AHPC_B_RIP_F8_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_D_RIP_F7_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_B_TuJ1_F5_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_TuJ1_F3_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_MAP2ab_F4_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_Ki67_F9_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_TuJ1_F4_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_C_Map2AB_F7_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_C_RIP_F1_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_B_TuJ1_F8_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_Map2AB_F4_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_C_Map2AB_F10_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_GFAP_F9_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_C_TuJ1_F9_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_B_Ki67_F8_Cy3_ND8_20x.tiff",
                                  "220909_GFP-AHPC_B_Ki67_F8_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_Ki67_F3_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_Map2AB_F4_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_Nestin_F1_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_A_GFAP_F9_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_Ki67_F8_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_Nestin_F10_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_C_TuJ1_F3_Cy3_ND1_20x.tiff",
                                  "220912_GFP-AHPC_A_Map2AB_F1_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_A_RIP_F1_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_A_TuJ1_F3_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_C_Map2AB_F8_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_A_Map2AB_F9_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_B_Map2AB_F1_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_Map2AB_F10_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_B_TuJ1_F1_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_B_GFAP_F4_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_D_GFAP_F3_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_Ki67_F4_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_Map2AB_F6_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_D_Ki67_F2_Cy3_ND8_20x.tiff",
                                  "220909_GFP-AHPC_A_GFAP_F7_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_B_GFAP_F6_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_A_TuJ1_F6_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_A_RIP_F2_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_C_TuJ1_F4_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_C_TuJ1_F10_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_B_GFAP_F10_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_A_TuJ1_F3_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_D_TuJ1_F8_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_C_Nestin_F1_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_A_Ki67_F3_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_Nestin_F2_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_B_GFAP_F2_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_C_GFAP_F5_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_Nestin_F10_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_C_Map2AB_F9_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_C_TuJ1_F1.2_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_RIP_F8_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_A_Ki67_F2_CY3_ND8_20x.tiff",
                                  "220912_GFP-AHPC_A_Map2AB_F6_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_B_Ki67_F9_Cy3_ND8_20x.tiff",
                                  "220912_GFP-AHPC_A_RIP_F4_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_B_Nestin_F8_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_D_GFAP_F6_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_GFAP_F9_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_D_GFAP_F2_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_C_Map2AB_F7_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_TuJ1_F10_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_A_GFAP_F7_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_D_MAP2ab_F4_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_C_GFAP_F7_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_A_Ki67_F2_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_A_TuJ1_F5_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_D_Ki67_F10_Cy3_ND8_20x.tiff",
                                  "220909_GFP-AHPC_D_Nestin_F6_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_C_Nestin_F2_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_C_RIP_F7_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_D_TuJ1_F3_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_A_GFAP_F9_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_B_Ki67_F10_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_TuJ1_F10_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_TuJ1_F9_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_A_Nestin_F6_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_B_Nestin_F3_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_D_Nestin_F7_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_Nestin_F6_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_GFAP_F8_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_D_Ki67_F8_Cy3_ND8_20x.tiff",
                                  "220909_GFP-AHPC_D_Ki67_F4_Cy3_ND8_20x.tiff",
                                  "220912_GFP-AHPC_C_Map2AB_F2_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_B_Map2AB_F10_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_C_Map2AB_F10_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_A_Map2AB_F8_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_A_GFAP_F10_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_A_GFAP_F4_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_B_GFAP_F3_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_A_Nestin_F9_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_B_Map2AB_F1_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_A_TuJ1_F9_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_D_TuJ1_F6_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_C_Map2AB_F5_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_A_Ki67_F9_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_A_Nestin_F2_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_A_GFAP_F10_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_Nestin_F8_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_B_Nestin_F8_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_A_Map2AB_F2_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_A_Map2AB_F8_Cy3_ND2_20x.tiff",
                                  "220909_GFP-AHPC_D_TuJ1_F8_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_A_GFAP_F6_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_TuJ1_F1_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_C_Ki67_F2_DAPI_ND1_20x.tiff",
                                  "220909_GFP-AHPC_D_Nestin_F6_DAPI_ND1_20x.tiff",
                                  "220912_GFP-AHPC_A_RIP_F6_Cy3_ND2_20x.tiff",
                                  "220912_GFP-AHPC_C_TuJ1_F1.1_DAPI_ND1_20x.tiff"]
test_images_filenames = ["220909_GFP-AHPC_D_MAP2ab_F7_Cy3_ND2_20x.tiff",
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
val_images_filenames = ["220912_GFP-AHPC_A_Nestin_F9_DAPI_ND1_20x.tiff",
                                "220912_GFP-AHPC_B_Nestin_F9_DAPI_ND1_20x.tiff",
                                "220912_GFP-AHPC_C_Map2AB_F8_Cy3_ND2_20x.tiff",
                                "220912_GFP-AHPC_B_Map2AB_F2_Cy3_ND2_20x.tiff",
                                "220912_GFP-AHPC_C_TuJ1_F4_Cy3_ND1_20x.tiff",
                                "220912_GFP-AHPC_A_Nestin_F10_DAPI_ND1_20x.tiff",
                                "220912_GFP-AHPC_B_Map2AB_F6_DAPI_ND1_20x.tiff",
                                "220909_GFP-AHPC_D_MAP2ab_F1_DAPI_ND1_20x.tiff",
                                "220912_GFP-AHPC_A_Map2AB_F1_DAPI_ND1_20x.tiff",
                                "220909_GFP-AHPC_D_Nestin_F9_Cy3_ND2_20x.tiff",
                                "220909_GFP-AHPC_B_Ki67_F6_Cy3_ND8_20x.tiff",
                                "220912_GFP-AHPC_B_Map2AB_F9_DAPI_ND1_20x.tiff",
                                "220909_GFP-AHPC_D_MAP2ab_F5_Cy3_ND2_20x.tiff",
                                "220912_GFP-AHPC_A_RIP_F7_Cy3_ND2_20x.tiff",
                                "220912_GFP-AHPC_A_TuJ1_F7_Cy3_ND2_20x.tiff",
                                "220912_GFP-AHPC_B_TuJ1_F1_Cy3_ND2_20x.tiff",
                                "220912_GFP-AHPC_C_RIP_F4_Cy3_ND2_20x.tiff",
                                "220909_GFP-AHPC_D_Ki67_F6_DAPI_ND1_20x.tiff",
                                "220909_GFP-AHPC_D_Nestin_F4_Cy3_ND2_20x.tiff",
                                "220912_GFP-AHPC_A_Map2AB_F4_DAPI_ND1_20x.tiff",
                                "220909_GFP-AHPC_A_GFAP_F4_Cy3_ND2_20x.tiff",
                                "220909_GFP-AHPC_C_GFAP_F4_DAPI_ND1_20x.tiff",
                                "220909_GFP-AHPC_D_GFAP_F9_DAPI_ND1_20x.tiff",
                                "220909_GFP-AHPC_D_TuJ1_F4_DAPI_ND1_20x.tiff",
                                "220909_GFP-AHPC_D_RIP_F6_Cy3_ND2_20x.tiff",
                                "220909_GFP-AHPC_D_TuJ1_F1_Cy3_ND2_20x.tiff",
                                "220909_GFP-AHPC_C_GFAP_F6_Cy3_ND2_20x.tiff",
                                "220912_GFP-AHPC_A_Nestin_F8_Cy3_ND2_20x.tiff",
                                "220909_GFP-AHPC_C_Nestin_F5_DAPI_ND1_20x.tiff",
                                "220909_GFP-AHPC_C_Nestin_F3_Cy3_ND2_20x.tiff",
                                "220909_GFP-AHPC_D_MAP2ab_F2_Cy3_ND2_20x.tiff",
                                "220912_GFP-AHPC_C_Map2AB_F9_DAPI_ND1_20x.tiff",
                                "220912_GFP-AHPC_C_RIP_F5_Cy3_ND2_20x.tiff",
                                "220909_GFP-AHPC_D_GFAP_F6_Cy3_ND2_20x.tiff",
                                "220909_GFP-AHPC_C_Ki67_F8_Cy3_ND8_20x.tiff",
                                "220909_GFP-AHPC_C_Ki67_F5_DAPI_ND1_20x.tiff",
                                "220912_GFP-AHPC_C_TuJ1_F3_DAPI_ND1_20x.tiff",
                                "220909_GFP-AHPC_D_Ki67_F6_Cy3_ND8_20x.tiff",
                                "220909_GFP-AHPC_B_Ki67_F10_Cy3_ND8_20x.tiff",
                                "220909_GFP-AHPC_A_Ki67_F7_DAPI_ND1_20x.tiff",
                                "220912_GFP-AHPC_B_Nestin_F2_DAPI_ND1_20x.tiff",
                                "220909_GFP-AHPC_C_Nestin_F8_DAPI_ND1_20x.tiff"]


#training configuration
start_step = 0
end_step = 2000
lr = 0.00001
momentum = 0.9
disp_interval = 500
log_interval = 250


#Tensorboard  config
use_tensorboard = False
save_exp_name = method + '_' + dataset_name + '_' + 'v1'
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------
rand_seed = 64678  
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


# load net
net = CrowdCounter()
network.weights_normal_init(net, dev=0.01)
net.cuda()
net.train()

params = list(net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:    
        exp_name = save_exp_name 
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

data_loader = ImageDataLoader(train_images_filenames, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
data_loader_val = ImageDataLoader(val_images_filenames, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)
best_mae = math.inf

for epoch in range(start_step, end_step+1):    
    step = -1
    train_loss = 0
    for blob in data_loader:                
        step = step + 1        
        im_data = blob['data']
        gt_data = blob['gt_density']
        density_map = net(im_data, gt_data)
        loss = net.loss
        train_loss += loss.data
        step_cnt += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % disp_interval == 0:            
            duration = t.toc(average=False)
            fps = step_cnt / duration
            gt_count = np.sum(gt_data)    
            density_map = density_map.data.cpu().numpy()
            et_count = np.sum(density_map)
            utils.save_results(im_data,gt_data,density_map, output_dir)
            log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f' % (epoch,
                step, 1./fps, gt_count,et_count)
            log_print(log_text, color='green', attrs=['bold'])
            re_cnt = True    
    
       
        if re_cnt:                                
            t.tic()
            re_cnt = False

    if (epoch % 2 == 0):
        save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method,dataset_name,epoch))
        network.save_net(save_name, net)     
        #calculate error on the validation dataset 
        mae,mse = evaluate_model(save_name, data_loader_val)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = '{}_{}_{}.h5'.format(method,dataset_name,epoch)
            network.save_net(os.path.join(output_dir,"best_model.h5"), net)
        log_text = 'EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (epoch,mae,mse)
        log_print(log_text, color='green', attrs=['bold'])
        log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, BEST MODEL: %s' % (best_mae,best_mse, best_model)
        log_print(log_text, color='green', attrs=['bold'])
        if use_tensorboard:
            exp.add_scalar_value('MAE', mae, step=epoch)
            exp.add_scalar_value('MSE', mse, step=epoch)
            exp.add_scalar_value('train_loss', train_loss/data_loader.get_num_samples(), step=epoch)
        
    

