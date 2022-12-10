import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2

def make_transform():
    base_transform = [

    ]

    train_transform= [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, 
        #                 border_mode=cv2.BORDER_CONSTANT, p=0.5), 
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.5),
        A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=0.5),
        A.RandomCrop(320, 320),
    ]

    train_transform.extend(base_transform)

    train_transform = A.Compose(train_transform)
    test_transform = A.Compose(base_transform)
    
    return train_transform, test_transform