# 항공사진을 이용한 토지피복지도 객체 분할 - AI CONNECT 모의경진대회

<br>

## 1. Data set

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


**mIoU**
- IoU(Intersection over Union)

<br>

- mIoU(Mean Intersection over Union)

<br>

## 3. Modeling

### 3-1. Augmentations
- **Use Albumentations library**
- HorizontalFlip
- VerticalFlip
- ShiftScaleRotate
- RandomBrightnessContrast
- CLAHE
- RandomCrop


### 3-2. Models
- Unet
- Unet++
- DeepLabV3+
- **Ensemble(best)**
    - Unet++_EfficientnetB2_Kfold3
    - Unet++_EfficientnetB3_Kfold1
    - Unet++_EfficientnetB3_Kfold3

### 3-3. Encoders
- **EfficientnetB2 ~ B4(best)**
- resnet50
- Mix Vision Transformer(mit_b3)

### 3-4. Losses
- **BCEWithLogitsLoss(best)**
- Soft BCEWithLogitsLoss
- DiceLoss
- TverskyLoss
- ForcalLoss
- LovaszLoss

### 3-5. Optimizers
- **adam(best)**
- adamw
- adamp

<br>

## 4. Inference

 - Test Time Augmentation
    - **Use ttach library**
    - HorizontalFlip
    - Rotate90
    - VerticalFlip
- Threshold: **0.4(best)**, 0.5, 0.6

<br>

## 5. Model Evaluation

**[5-1. Encoders]**
| Encoders      | Loss          | mIoU  |
| ------------- |:-------------:| -----:|
| EfficientnetB2      | 0.06575      |   0.7508 |
| EfficientnetB3   | 0.06269      |    0.7817 |
| EfficientnetB4     | 0.06193      |    0.7677 |
| Resnet50    | 0.08719      |    0.6968 |
| Mit_b3     | 0.08036      |    0.7265 |

(Model: Unet++)

<br>

**[5-2. Losses]**
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

**[5-3. Models]**
| Models        | Loss          | mIoU  |
| ------------- |:-------------:| -----:|
| Unet      | 0.0749      |   0.7193 |
| Unet++    | 0.06269      |    0.7817 |
| DeepLabV3+     | 0.06782      |    0.7403 |
| Ensemble    | 0.04267      |    0.8163 |

<br>

## 6. Result
- mIoU Final Score: 0.7263
- 2등 / 134팀

<br>

## 7. Review
- Augmentation으로 모델 성능이 대폭 향상하는 것을 통해 데이터의 중요성을 다시 한 번 느끼게 되었다.
- Ensemble을 통해 score가 많이 상승되었는데, 성능이 좋은 모델만 사용했을 때 보다 성능이 낮은 모델도 함께 ensemble했을 때 더 좋은 성능을 보인다는 것을 알게되었다.
- TTA를 활용하여 score를 유의미하게 올릴 수 있었다. 시간 부족으로 여러 방법의 TTA를 사용하지 못한 것이 아쉽다. 추후 여러가지 Voting방법을 사용해 본다면 더 좋은 점수를 얻을 수 있을 것 같다.
- 본격적으로 모델 학습을 하기 전, Base Line과 가벼운 모델을 사용하며 여러 실험을 했던 것이 많이 도움이 되었다.