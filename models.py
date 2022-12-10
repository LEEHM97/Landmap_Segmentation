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


class UnetPlusPlus_B3(pl.LightningModule):
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
    
    
class UnetPlusPlus_B2(pl.LightningModule):
    def __init__(self, args=None, optimizer='adam', scheduler='reducelr'):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name='efficientnet-b2',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
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
    
    
class UnetPlusPlus_B4(pl.LightningModule):
    def __init__(self, args=None, optimizer='adam', scheduler='reducelr'):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name='efficientnet-b4',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
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
    
    
class DeepLabV3Plus(pl.LightningModule):
    def __init__(self, args=None, optimizer='adam', scheduler='reducelr'):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
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
       
class Ensemble(pl.LightningModule):
    def __init__(self, model1, model2, model3, args=None, optimizer='adam', scheduler='reducelr'):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
                
        self.model1.model.segmentation_head = nn.Identity()
        self.model2.model.segmentation_head = nn.Identity()
        self.model3.model.segmentation_head = nn.Identity()
        
        self.cv = nn.Conv2d(48, 1, 1)
        # self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        
        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        x1 = self.model1(x.clone())
        x2 = self.model2(x)
        x3 = self.model3(x)
        # x2 = self.up(x2)

        x = torch.cat((x1, x2, x3), dim=1)
        output = self.cv(x)
        
        return output

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

        outputs = self.forward(image)
        loss = self.criterion(outputs, mask.unsqueeze(1).float())
        jaccard_index_value = binary_jaccard_index(torch.sigmoid(outputs), mask.unsqueeze(0).permute(1, 0, 2, 3))

        self.log('train/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('train/jaccard_index_value', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)
        
        return {"loss": loss, "jaccard_index": jaccard_index_value}

    def validation_step(self, val_batch, batch_idx):
        image, mask = val_batch
        mask = mask.long()

        outputs = self.forward(image)
        loss = self.criterion(outputs, mask.unsqueeze(1).float())
        jaccard_index_value = binary_jaccard_index(torch.sigmoid(outputs), mask.unsqueeze(0).permute(1, 0, 2, 3))

        self.log('val/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('val/jaccard_index_value', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)
        
        return {"loss": loss, "jaccard_index": jaccard_index_value} 
    
    
class UnetPlusPlus_Inception(pl.LightningModule):
    def __init__(self, args=None, optimizer='adam', scheduler='reducelr'):
        super().__init__()
        self.model = smp.Unet(
            encoder_name='mit_b3',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
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
        loss = self.criterion(outputs, mask.unsqueeze(1).float())
        jaccard_index_value = binary_jaccard_index(torch.sigmoid(outputs), mask.unsqueeze(0).permute(1, 0, 2, 3))

        self.log('val/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('val/jaccard_index_value', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)
        
        return {"loss": loss, "jaccard_index": jaccard_index_value} 