import os
import torch
import wandb
import glob
import natsort

import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import numpy as np
import pandas as pd

from tqdm import tqdm
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from datasets import SegmentationDataset, TestDataset
from transforms import make_transform
from models import Unet, UnetPlusPlus, DeepLabV3Plus, Ensemble

def mask_to_rle(mask):
    flatten_mask = mask.flatten()
    if flatten_mask.max() == 0:
        return f'0 {len(flatten_mask)}'
    idx = np.where(flatten_mask!=0)[0]
    steps = idx[1:]-idx[:-1]
    new_coord = []
    step_idx = np.where(np.array(steps)!=1)[0]
    start = np.append(idx[0], idx[step_idx+1])
    end = np.append(idx[step_idx], idx[-1])
    length = end - start + 1
    for i in range(len(start)):
        new_coord.append(start[i])
        new_coord.append(length[i])
    new_coord_str = ' '.join(map(str, new_coord))
    return new_coord_str


    
# 프로젝트 경로
PROJECT_DIR = './'
os.chdir(PROJECT_DIR)

#데이터 경로
DATA_DIR = os.path.join(PROJECT_DIR, 'DATA') # 모든 데이터가 들어있는 폴더 경로
TEST_DIR = os.path.join(DATA_DIR, 'test') # 테스트 데이터가 들어있는 폴더 경로
TEST_IMG_DIR = os.path.join(TEST_DIR, 'images') # 테스트 이미지가 들어있는 폴더 경로
TEST_CSV_FILE = os.path.join(TEST_DIR, 'testdf.csv') # 테스트 이미지 이름이 들어있는 CSV 경로
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

testdf = pd.read_csv(TEST_CSV_FILE)
test_dataset = TestDataset(testdf, TEST_IMG_DIR)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1,shuffle=False)

model1 = UnetPlusPlus()
model2 = DeepLabV3Plus()

model1 = model1.load_from_checkpoint('./Models/unet++_b3.ckpt')
model2 = model2.load_from_checkpoint('./Models/DeepLabV3+_crop_fold01_val/jaccard_index_value=0.7248.ckpt')

model1.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model2.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


model = Ensemble(model1=model1, model2=model2)
model = model.load_from_checkpoint('./Models/ensembleV1_fold01_val/jaccard_index_value=0.7952.ckpt', model1=model1, model2=model2)
model.to(DEVICE)
model.eval()

file_list = [] # 이미지 이름 저장할 리스트
pred_list = [] # 마스크 저장할 리스트
class_list = [] # 클래스 이름 저장할 리스트 ('building')

model.eval()
with torch.no_grad():
    for batch_index, (image,imname) in tqdm(enumerate(test_loader)):
        image = image.to(DEVICE)
        logit_mask = model(image)
        pred_mask = torch.sigmoid(logit_mask) # logit 값을 probability score로 변경
        pred_mask = (pred_mask > 0.6) * 1.0 # 0.5 이상 확률 가진 픽셀값 1로 변환
        pred_rle = mask_to_rle(pred_mask.detach().cpu().squeeze(0)) # 마스크를 RLE 형태로 변경
        pred_list.append(pred_rle)
        file_list.append(imname[0])
        class_list.append("building")
        
        
# 예측 결과 데이터프레임 만들기
results = pd.DataFrame({'img_id':file_list,'class':class_list,'prediction':pred_list})

# sample_submission.csv와 같은 형태로 변형
sampledf = pd.read_csv('./DATA/test/sample_submission.csv')
sorter = list(sampledf['img_id'])
results = results.set_index('img_id')
results = results.loc[sorter].reset_index()
                       
# 결과 저장
results.to_csv('./Results/ensembleV1_6.csv', index=False)