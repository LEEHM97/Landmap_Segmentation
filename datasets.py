import cv2
import torch

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, augmentations):
        self.images = images
        self.masks = masks
        self.augmentations = augmentations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = cv2.imread(self.images[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks[item], 0)
        # mask = np.expand_dims(mask, axis=-1)

        if self.augmentations:
            transformed = self.augmentations(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'] 
        
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        # mask = np.transpose(mask, (2,0,1)).astype(np.float32)

        image = torch.Tensor(image) / 255.0
        mask = torch.round(torch.Tensor(mask)/255.0)

        return image, mask