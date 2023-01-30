import os
import random
import cv2
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
import Config
from dataset.dataset import IDCIA
from model.CellQuant import CellQuant
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins.environments import SLURMEnvironment


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CellDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage=None):
        # Get the image file names and make sure they are valid images.
        images_filenames = list(sorted(os.listdir(Config.IMAGES_DIRECTORY)))
        correct_images_filenames = [i for i in images_filenames if cv2.imread(
            os.path.join(Config.IMAGES_DIRECTORY, i)) is not None]
        # Shuffle the images list before split. Using a random seed.
        random.seed(42)
        random.shuffle(correct_images_filenames)
        # Take a smaller portion of the original dataset
        sub = int(len(correct_images_filenames) * 1)
        correct_images_filenames = correct_images_filenames[:sub]
        # Perform train valid test split of 800:150:50
        train_size = int(len(correct_images_filenames) * .6)
        test_size = int(len(correct_images_filenames) * .2)
        train_images_filenames = correct_images_filenames[:train_size]
        val_images_filenames = correct_images_filenames[train_size:-test_size]
        test_images_filenames = images_filenames[-test_size:]
        print(len(train_images_filenames), len(val_images_filenames), len(test_images_filenames))
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

        self.train_data = IDCIA(train_images_filenames, Config.IMAGES_DIRECTORY,
                                Config.MASKS_DIRECTORY, transform=Config.train_transform)
        self.valid_data = IDCIA(val_images_filenames, Config.IMAGES_DIRECTORY,
                                Config.MASKS_DIRECTORY, transform=Config.val_transform)
        self.test_data = IDCIA(test_images_filenames, Config.IMAGES_DIRECTORY,
                               Config.MASKS_DIRECTORY, transform=Config.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=Config.BATCH_SIZE, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=Config.BATCH_SIZE, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=Config.BATCH_SIZE, shuffle=False)


for i in range(Config.experiments):
    print(f"Running Experiment {i + 1} out of {Config.experiments}")
    ployp_data = CellDataModule()
    ployp_data.setup()
    counter = CellQuant()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator='auto', callbacks=[TQDMProgressBar(), lr_monitor], max_epochs=200,
                         log_every_n_steps=7, detect_anomaly=True,plugins=[SLURMEnvironment(auto_requeue=False)])
    trainer.fit(counter, ployp_data)
    trainer.test(counter, ployp_data.test_dataloader())
