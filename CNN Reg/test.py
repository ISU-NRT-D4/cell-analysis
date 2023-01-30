import torchvision.models as models
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy
import Config
from dataset.dataset import IDCIA
from torch.utils.data import Dataset,DataLoader
import random
from pytorch_lightning.callbacks import TQDMProgressBar
import json
from model.CellQuant import CellQuant
import os
import cv2


class CellDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
    def setup(self,stage=None):
        # Get the image file names and make sure they are valid images.
        images_filenames = list(sorted(os.listdir(Config.IMAGES_DIRECTORY)))
        correct_images_filenames = [i for i in images_filenames if cv2.imread(
            os.path.join(Config.IMAGES_DIRECTORY, i)) is not None]
        # Shuffle the images list before split. Using a random seed.
        random.seed(42)
        random.shuffle(correct_images_filenames)
        #Take a smaller portion of the original dataset
        sub=int(len(correct_images_filenames)*1)
        correct_images_filenames=correct_images_filenames[:sub]
        # Perform train valid test split of 800:150:50
        train_size = int(len(correct_images_filenames)*.6)
        test_size = int(len(correct_images_filenames)*.2)
        train_images_filenames = correct_images_filenames[:train_size]
        val_images_filenames = correct_images_filenames[train_size:-test_size]
        test_images_filenames = images_filenames[-test_size:]
        print(len(train_images_filenames), len(val_images_filenames), len(test_images_filenames))

        train_set = json.dumps(train_images_filenames)
        valid_set = json.dumps(val_images_filenames)
        test_set = json.dumps(test_images_filenames)



        self.train_data = IDCIA(train_images_filenames, Config.IMAGES_DIRECTORY,
                        Config.MASKS_DIRECTORY, transform=Config.train_transform)
        self.valid_data = IDCIA(val_images_filenames, Config.IMAGES_DIRECTORY,
                        Config.MASKS_DIRECTORY, transform=Config.val_transform)
        self.test_data = IDCIA(test_images_filenames, Config.IMAGES_DIRECTORY,
                        Config.MASKS_DIRECTORY, transform=Config.test_transform)                        
    def train_dataloader(self):

        return DataLoader(self.train_data,batch_size=Config.BATCH_SIZE,shuffle = False)

    def val_dataloader(self):  

        return DataLoader(self.valid_data,batch_size=Config.BATCH_SIZE,shuffle = False)

    def test_dataloader(self):

        return DataLoader(self.test_data,shuffle = False)


ployp_data=CellDataModule()

ployp_data.setup()

model = CellQuant.load_from_checkpoint("lightning_logs/version_2/checkpoints/epoch=199-step=2000.ckpt")


trainer = pl.Trainer(accelerator='auto', callbacks=[TQDMProgressBar(refresh_rate=7)], max_epochs=200, log_every_n_steps=7, detect_anomaly=True)  # for Colab: set refresh rate to 20 instead of 10 to avoid freezing

test_result = trainer.test(model, ployp_data.test_dataloader())

print(test_result)

# predictions=trainer.predict(model, ployp_data.test_dataloader())







