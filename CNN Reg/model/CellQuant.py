import torchvision.models as models
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy
import Config


class CellQuant(pl.LightningModule):
    def __init__(self, lr=Config.lr):
        super().__init__()
        self.save_hyperparameters()

        self.model = models.vgg16_bn(pretrained=True)

        num_filters = self.model.classifier[0].in_features

        for param in self.model.parameters():
            param.requires_grad = False

        # use the pretrained model to predict the count
        self.model.classifier = nn.Sequential(
            nn.Linear(num_filters, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, input_data):
        preds = self.model(input_data)

        return preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)

        loss = nn.L1Loss()(preds, y)

        self.log('train_loss', loss)  # lightning detaches your loss graph and uses its value

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)

        loss = nn.L1Loss()(preds, y)
        self.log('validation_loss', loss)  # lightning detaches your loss graph and uses its value

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)

        loss = nn.L1Loss()(preds, y)


        self.log('test_loss', loss)  # lightning detaches your loss graph and uses its value

    def predict_step(self, batch, batch_idx):
        x, y = batch

        pred = self.model(x)
        print(f"\n Label: {y.item()} and Prediction: {pred.item()} \n")
        return pred

    def configure_optimizers(self):
        # return optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                                        threshold=0.0001, threshold_mode='abs',verbose=True),
                "interval": "epoch", # This has been added later on.
                "monitor": "validation_loss",
            },
        }
