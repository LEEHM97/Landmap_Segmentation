import os
import torch
import wandb
import glob
import natsort

import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from datasets import SegmentationDataset
from transforms import make_transform
from models import Unet, UnetPlusPlus, DeepLabV3Plus, Ensemble

train_data_dir = "./DATA/train"
test_data_dir = "./DATA/test"

## Ensemble
model1 = UnetPlusPlus()
model2 = DeepLabV3Plus()

model1 = model1.load_from_checkpoint('./Models/unet++_b3.ckpt')
model2 = model2.load_from_checkpoint('./Models/DeepLabV3+_crop_fold01_val/jaccard_index_value=0.7248.ckpt')

model1.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model2.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

if __name__ == "__main__":
    pl.seed_everything(42)
    
    train_images = np.array(natsort.natsorted(glob.glob(os.path.join(train_data_dir, "images", "*"))))
    train_masks = np.array(natsort.natsorted(glob.glob(os.path.join(train_data_dir, "masks", "*"))))

    test_images = np.array(natsort.natsorted(glob.glob(os.path.join(test_data_dir, "images", "*"))))
    
    kf = KFold(n_splits=3)
    for idx, (train_index, val_index) in enumerate(kf.split(X=train_images)):
        wandb_logger = WandbLogger(project="Landmap_Segmentation", name=f'ensembleV1_fold{idx + 1:02d}', entity="leehm")

        checkpoint_callback = ModelCheckpoint(
                monitor="val/jaccard_index_value",
                dirpath="./Models",
                filename=f"ensembleV1_fold{idx + 1:02d}_" + "{val/jaccard_index_value:.4f}",
                save_top_k=3,
                mode="max",
                # save_weights_only=True
            )

        early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=True,
                                                mode="min")
        model = Ensemble(model1=model1, model2=model2)

        train_transform, test_transform = make_transform()

        train_ds = SegmentationDataset(train_images[train_index], train_masks[train_index], train_transform)
        train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=16, num_workers=4, shuffle=True, drop_last=True)

        val_ds = SegmentationDataset(train_images[val_index], train_masks[val_index], train_transform)
        val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=16, num_workers=4)


        trainer = pl.Trainer(accelerator='gpu',
                        devices=1,
                        precision=16,
                        max_epochs=50,
                        log_every_n_steps=1,
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback])
                        
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
        wandb.finish()