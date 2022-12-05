import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from torchmetrics.functional.classification import binary_jaccard_index 

class Unet(pl.LightningModule):
    def __init__(self, args=None, optimizer='adam', scheduler='reducelr'):
        super().__init__()
        self.model = smp.Unet(
            encoder_name='efficientnet-b3',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        
        )
        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)

        if self.scheduler == "reducelr":
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode="max", verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}

        elif self.args.scheduler == "cosineanneal":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5,
                                                                 last_epoch=-1, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, train_batch, batch_idx):
        image, mask = train_batch
        mask = mask.long()

        outputs = self.model(image)
        loss = self.criterion(outputs, mask.unsqueeze(1).float())
        jaccard_index_value = binary_jaccard_index(torch.sigmoid(outputs), mask.unsqueeze(0).permute(1, 0, 2, 3))

        self.log('train/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('train/jaccard_index_value', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)
        
        return {"loss": loss, "jaccard_index": jaccard_index_value}

    def validation_step(self, val_batch, batch_idx):
        image, mask = val_batch
        mask = mask.long()

        outputs = self.model(image)

        # loss = self.criterion(outputs, mask.unsqueeze(1).float())
        loss = self.criterion(outputs, mask.unsqueeze(1).float())
        jaccard_index_value = binary_jaccard_index(torch.sigmoid(outputs), mask.unsqueeze(0).permute(1, 0, 2, 3))

        self.log('val/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('val/jaccard_index_value', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)
        
        return {"loss": loss, "jaccard_index": jaccard_index_value}


class UnetPlusPlus(pl.LightningModule):
    def __init__(self, args=None, optimizer='adam', scheduler='reducelr'):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name='efficientnet-b3',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        
        )
        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        x = self.model(x)
        return x