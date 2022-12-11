# 항공사진을 이용한 토지피복지도 객체 분할 - AI CONNECT 모의경진대회

<br>

## 1. Data set
---
- Input: 512X512 크기의 항공/위성사진
- Target: 512X512 크기의 마스크
- n_train: 3930
- n_test: 1303
- Data directory structure:
    ```bash
    DATA/
    ├── train/
    │         ├── traindf.csv  
    │         ├── images/
    │         │        ├── xxx.png
    │         │        ├── yyy.png
    │         │        ├── zzz.png
    │         │        └── ...  
    │         └── masks/
    │                      ├── xxx.png
    │                      ├── yyy.png
    │                      ├── zzz.png
    │                      └── ...  
    └── test/
                ├── sample_submission.csv
                ├── testdf.csv
                └── images/
                            ├──  aaa.png
                            ├──  bbb.png  
                            └──   ...
    ```

<br>

## 2. Evaluation Metrics
---

**mIoU**
- IoU(Intersection over Union)

<br>

- mIoU(Mean Intersection over Union)

<br>

## 3. Modeling
---
### 3-1. Augmentations
- **Use Albumentations library**
- HorizontalFlip
- VerticalFlip
- ShiftScaleRotate
- RandomBrightnessContrast
- CLAHE
- RandomCrop

<br>

### 3-2. Models
- Unet
- Unet++
- DeepLabV3+
- **Ensemble(best)**
    - Unet++_EfficientnetB2_Kfold3
    - Unet++_EfficientnetB3_Kfold1
    - Unet++_EfficientnetB3_Kfold3

<br>

### 3-3. Encoders
- **EfficientnetB2 ~ B4(best)**
- resnet50
- Mix Vision Transformer(mit_b3)

<br>

### 3-4. Losses
- **BCEWithLogitsLoss(best)**
- Soft BCEWithLogitsLoss
- DiceLoss
- TverskyLoss
- ForcalLoss
- LovaszLoss

<br>

### 3-5. Optimizers
- **adam(best)**
- adamw
- adamp

<br>

## 4. Inference
---
 - Test Time Augmentation
    - **Use ttach library**
    - HorizontalFlip
    - Rotate90
    - VerticalFlip
- Threshold: **0.4(best)**, 0.5, 0.6

<br>

## 5. Model Evaluation
---

| Encoders      | Loss          | mIoU  |
| ------------- |:-------------:| -----:|
| EfficientnetB2      | 0.06575      |   0.7508 |
| EfficientnetB3   | 0.06269      |    0.7817 |
| EfficientnetB4     | 0.06193      |    0.7677 |
| Resnet50    | 0.08719      |    0.6968 |
| Mit_b3     | 0.08036      |    0.7265 |
(Model: Unet++)

<br>

| Losses        | Loss          | mIoU  |
| ------------- |:-------------:| -----:|
| DiceLoss      | 0.1747      |   0.7081 |
| TverskyLoss   | 0.1524      |    0.7385 |
| FocalLoss     | 0.01949      |    0.7356 |
| LovaszLoss    | 0.581      |    0.7154 |
| SoftBCEWithLogitsLoss     | 0.06988      |    0.7451 |
| BCEWithLogitsLoss     | 0.06269      |    0.7817 |
(Model: Unet++, Encoder: EfficientnetB3)

<br>

| Models        | Loss          | mIoU  |
| ------------- |:-------------:| -----:|
| Unet      | 0.0749      |   0.7193 |
| Unet++    | 0.06269      |    0.7817 |
| DeepLabV3+     | 0.06782      |    0.7403 |
| Ensemble    | 0.04267      |    0.8163 |