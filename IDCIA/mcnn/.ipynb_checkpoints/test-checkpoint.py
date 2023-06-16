import os
import torch
import numpy as np

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

data_path =  './data/original/shanghaitech/part_B_final/test_data/images/'
gt_path = './data/original/shanghaitech/part_B_final/test_data/ground_truth_csv/'
model_path = 'saved_models/best_model.h5'

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


output_dir = './output/'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


net = CrowdCounter()
      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()
mae = 0.0
mse = 0.0

#load test data
data_loader = ImageDataLoader(test_images_filenames, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

for blob in data_loader:                        
    im_data = blob['data']
    gt_data = blob['gt_density']
    density_map = net(im_data, gt_data)
    density_map = density_map.data.cpu().numpy()
    gt_count = np.sum(gt_data)
    et_count = np.sum(density_map)
    mae += abs(gt_count-et_count)
    mse += ((gt_count-et_count)*(gt_count-et_count))
    if vis:
        utils.display_results(im_data, gt_data, density_map)
    if save_output:
        utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')
        
mae = mae/data_loader.get_num_samples()
mse = np.sqrt(mse/data_loader.get_num_samples())
print ('\nMAE: %0.2f, MSE: %0.2f' % (mae,mse))

f = open(file_results, 'w') 
f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
f.close()