#!/usr/bin/env python
# coding: utf-8

# # [Case 4-2] 품질 검증을 위한 이미지 분류_Rev.01

# ---

# 
# ## 프로젝트 목표
# ---
# - 이미지 처리 인공지능 시스템의 전반적인 이해.
# - **ResNet 기반 이미지 분류 모델 구현.**
# - 이미지 데이터 전처리 작업.
# - ResNet 모델 하이퍼파라미터 튜닝 기법.
# 

# ## 프로젝트 목차
# ---
# 
# 1. **데이터 읽기:** Local에 저장되어 있는 Kaggle 데이터를 불러오고 이미지 확인
# 
# 2. **데이터 전처리:** 모델에 필요한 입력 형태로 데이터 처리
# 
# 3. **ResNet 모델 로드  :** ResNet 모델 불러오기
# 
# 4. **하이퍼 파라미터 설정 및 컴파일 :** 올바른 하이퍼 파라미터 설정
# 
# 5. **ResNet 모델 학습 수행:** ResNet 모델을 통한 학습 수행, 평가 및 예측 수행
# 
# 6. **제출:** 예측한 결과를 제출한 후 채점 결과를 확인
# 
# 

# ## 데이터 출처
# ---
# http://https://www.kaggle.com/christianvorhemus/industrial-quality-control-of-packages
# 

# ## 프로젝트 개요
# ---
# 
# **데이터:** 가상 생산 라인에 있는 상자를 상단과 측면에서 찍은 이미지와 그 상자의 손상 여부 
# 
# **가정:** 손상된 상자는 특정한 시각적인 패턴이 있음
# 
# **목표:** 결함이 있는 상자가 발송되기 전에 공정에서 제외하도록 하자!
# 
# **제한사항:** 극소량의 학습 데이터 제공(클래스당 100개)
# 
#  ![00_example1](04_02_file/00_example1.png)
# 
# 

# #### Task 1
# 상자를 **상단에서 찍은 이미지와 측면에서 찍은 이미지 모두 사용**하여
# 
# 손상 여부를 판단할 수 있을까 ?
# 
#  ![00_example2](04_02_file/00_example2.png)

# #### Task 2
# 상자를 **상단에서 찍은 이미지만 사용**하여
# 
# 손상 여부를 판단할 수 있을까 ?
# 
#  ![00_example3](04_02_file/00_example3.png)

# ## 결과 요약
# 
# 각 파라미터 조건에 따른 모델 평가 결과는 아래와 같다.
# 
# |Run|Epochs|Scaler|criterion|frac|optimizer|lit|min_leaf|criterion|n_trees|Classifier_n_neighbors|학습_n_neighbors|epoch|변수|lr|class0|정확도|
# |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# |01|1||||RMSprop|||||||||||0.475000|
# |02|1||||SGD|||||||||||0.475000|
# |03|1||||Adam|||||||||||0.525000|
# |04|10||||Adam|||||||||||0.525000|
# 
# accuracy 측정 결과 정확도가 0.55 이상이면 100점
# 
# 목표 성능을 달성하기 위해서 아래 어떤 방법을 활용하여도 좋습니다.
# - 전처리
#     - IMG_SIZE
#     - SCALING 조정
# - 모델 수정
#     - KERNEL_SIZE
#     - CHANNEL_SIZE
#     - FC_SIZE
# - 학습 하이퍼 파라미터
#     - EPOCH_NUM
#     - BATCH_SIZE
#     - LEARNING_RATE
#     - OPTIMIZER 조정
# - 과적합 억제 및 초기화 기법
#     - Dropout
#     - Weight decay
#     - Weight Initialization 활용 등

# ## 1. 데이터 읽기
# 

# ### 1.1 라이브러리 불러오기

# In[ ]:


import os
import numpy as np

# 전처리를 위한 라이브러리
import random # 데이터 셔플링을 위한 라이브러리
from PIL import Image # 이미지 전처리 라이브러리

# 시각화 및 학습 로그를 위한 라이브러리
import matplotlib.pyplot as plt # 시각화 라이브러리

# 기타 라이브러리
from tqdm import tqdm # 학습 프로세스 시각화 라이브러리
from utils import recall, precision # 성능 평가 코드


# In[ ]:


# 모델 재현성을 위한 랜덤시드 설정
random.seed(42)
np.random.seed(42)


# ### 1.2 데이터 불러오기
# ---
# `손상된 상자` 데이터를 이미지 처리 라이브러리 PIL을 통해 모두 불러옵니다.
# 
# 

# In[ ]:


DAMAGED_DATA_DIR = '/mnt/data/chapter_4/damaged' #손상된 상자 데이터 경로
INTACT_DATA_DIR = '/mnt/data/chapter_4/intact' #손상되지 않은 상자 데이터 경로

damaged_side_path = os.path.join(DAMAGED_DATA_DIR, 'side') #손상된 상자 - 측면 촬영 데이터 경로
damaged_top_path = os.path.join(DAMAGED_DATA_DIR, 'top') #손상된 상자 - 상단 촬영 데이터 경로

intact_side_path = os.path.join(INTACT_DATA_DIR, 'side') #손상되지 않은 상자 - 측면 촬영 데이터 경로
intact_top_path = os.path.join(INTACT_DATA_DIR, 'top') #손상되지 않은 상자 - 상단 촬영 데이터 경로


# In[ ]:


# 주어진 데이터 경로로부터 각각의 이미지 파일을 읽어 dictionary (key: id, value: image)형태로 반환합니다.
def get_raw_data(path):
    raw_data = {}
    for item in os.listdir(path): # 파일명 리스트
        _id = item.split('_')[0] # 파일명에서 id만 추출
        _loc = os.path.join(path, item) # 디렉터리 경로와 파일명을 조인
        raw_data[_id] = np.array(Image.open(_loc)) # 각각의 이미지 경로로 부터 이미지를 읽어드려 행렬 (numpy) 형태로 저장
    return raw_data


# In[ ]:


# Raw data 불러오기 (손상된 박스 데이터)
damaged_side_data = get_raw_data(damaged_side_path)
damaged_top_data = get_raw_data(damaged_top_path)

# Raw data 불러오기 (손상되지 않은 박스 데이터)
intact_side_data = get_raw_data(intact_side_path)
intact_top_data = get_raw_data(intact_top_path)


# ## 2. 데이터 정제
# ---
# 
# 불러온 `손상된 상자`데이터에 대해서 전처리를 진행합니다.

# 
# #### 2.1 태스크에 따른 데이터 설정 
# ---
# 목표하고자 하는 태스크가 `Task 1`이면, 상단과 측면 이미지 모두 이용합니다.
# 
# -> 상단 이미지와 측면 이미지를 위아래로 붙이는 전처리 과정을 진행합니다.
# 
#  ![02_preprocessing1](04_02_file/02_preprocessing1.png)
# 
# 반면, 목표하고자 하는 태스크가 `Task 2`이면, 상단 이미지만 이용합니다.

# In[ ]:


# 전처리 관련 하이퍼파라미터를 설정합니다.

USE_TOP = False # 상단이미지 사용여부
USE_SIDE = True # 측면이미지 사용여부

IMG_SIZE = 224
BATCH_SIZE = 4
SCALING = 'Standard'


# In[ ]:


total_dataset = []

# 행렬화 부분
## 상단, 측면 둘 다 사용할 때는 이미지를 붙인다
if USE_TOP is True and USE_SIDE is True:
    for sample in damaged_top_data:
        top_data = damaged_top_data[sample]
        side_data = damaged_side_data[sample]
        image = np.concatenate((top_data, side_data), axis=0) # 상단 이미지와, 측면 이미지를 0번 축 기준으로 연결합니다.
        total_dataset.append((image, 0)) # 손상된 이미지의 레이블 -> 0

    for sample in intact_top_data:
        top_data = intact_top_data[sample]
        side_data = intact_side_data[sample]
        image = np.concatenate((top_data, side_data), axis=0)
        total_dataset.append((image, 1)) # 손상되지 않은 이미지의 레이블 -> 1

## 상단만 사용할 경우
elif USE_TOP is True and USE_SIDE is False:
    for sample in damaged_top_data:
        total_dataset.append((damaged_top_data[sample], [1,0]))

    for sample in intact_top_data:
        total_dataset.append((intact_top_data[sample], [0,1]))

## 측면만 사용할 경우
elif USE_TOP is False and USE_SIDE is True:
    for sample in damaged_side_data:
         total_dataset.append((damaged_side_data[sample], [1,0]))


    for sample in intact_side_data:
        total_dataset.append((intact_side_data[sample], [0,1]))


else:
    raise Exception
    

random.shuffle(total_dataset) # 데이터의 순서를 랜덤하게 섞어줍니다.


# In[ ]:


plt.imshow(total_dataset[0][0]) # 0번째 데이터의 이미지를 출력해봅니다.


# 
# #### 2.2 학습 / 검증 / 테스트 데이터 분할
# ---
# 
# 전체 데이터셋을 학습용 (Training), 검증용 (Validation), 테스트용 (Test) 데이터로 분할합니다.
# 
#  ![02_preprocessing3](04_02_file/02_preprocessing3.png)

# In[ ]:


# 훈련데이터, 검증데이터, 테스트데이터로 분할
train_idx = int(len(total_dataset) * 0.7) # 전체 데이터의 70%에 해당하는 지점의 index
val_idx = int(len(total_dataset) * 0.1) # 전체 데이터의 10%에 해당하는 지점의 index

train_dataset = total_dataset[:train_idx] # 전체 데이터의 70%를 훈련데이터로 사용합니다.
val_dataset = total_dataset[train_idx:train_idx + val_idx] # 전체 데이터의 10%를 검증 데이터로 사용합니다.
test_dataset = total_dataset[train_idx + val_idx:] # 전체 데이터의 20%를 테스트 데이터로 사용합니다.


# 
# #### 2.3 데이터를 텐서플로 형식에 맞게 변화 (formatting)
# ---
# 
# 데이터셋을 **리사이즈** 및 **스케일링**하고 **텐서플로우** 기반 딥러닝 모델에 입력될 수 있는 형식으로 **데이터 형식**을 전환합니다.
# 
# **Resize**
# 
#  ![02_preprocessing2](04_02_file/02_preprocessing2.png)
#  
#  
#  **Standard Style**
#  
#  **(x / 127.5) - 1**
#  
#   e.g., 
#  
#  (**255** / 127.5) - 1 = **1**       
#  
#  (**0** / 127.5) - 1 = **-1**
#  
#  **Min-Max Style**
#  
#   **x / 255.0**
#   
#  e.g., 
#  
#  **255** / 255.0 = **1**       
#  
#  **0** / 255.0 = **0**

# In[ ]:


# 텐서플로우 프레임워크
import tensorflow as tf 
tf.random.set_seed(42)


# In[ ]:


def format_dataset(dataset, resize=None):
    '''
    :param dataset: 데이터셋
    :param resize: resize 크기(e.g. 120, 180, 224)
    :return:
    '''
    image_arr = list()
    label_arr = list()
    
    for image, label in tqdm(dataset):
        image = np.array(image)
        label = label
        image_arr.append(image)
        label_arr.append(label)
        
    image_arr = np.array(image_arr, dtype=np.float32) # 이미지 행렬 리스트를 numpy 실수형 행렬로 변환
    image_arr = tf.cast(image_arr, tf.float32) # numpy 실수형 행렬을 텐서플로우 실수형 행렬로 변환
    label_arr = np.array(label_arr, np.int32) # 레이블 리스트를 numpy 정수형 행렬로 병환
    if SCALING == 'MinMax':
        image_arr = (image_arr / 255.0) # Min-Max Scaling
    elif SCALING == 'Standard':
        image_arr = (image_arr / 127.5) - 1 # 데이터 Normalization
    if resize is not None:
        image_arr = tf.image.resize(image_arr, (resize, resize)) # 이미지 리사이즈
    tf_dataset = tf.data.Dataset.from_tensor_slices((image_arr, label_arr)) # 텐서를 슬라이스하여 배치 단위로 쪼개기위한 함수
    return tf_dataset


# In[ ]:


tf_train_dataset = format_dataset(train_dataset, resize=IMG_SIZE)
tf_val_dataset = format_dataset(val_dataset, resize=IMG_SIZE)
tf_test_dataset = format_dataset(test_dataset, resize=IMG_SIZE)


# In[ ]:


for image_batch, label_batch in tf_train_dataset.take(1):
    pass


# In[ ]:


plt.imshow(image_batch)


# #### 2.4 데이터를 딥러닝 모델에 입력할 때의 배치사이즈를 설정
# ---
# 데이터는 딥러닝 모델에 들어갈 때 일정 batch 개수만큼 모델에 입력됩니다. 
# 

# In[ ]:


SHUFFLE_BUFFER_SIZE = 100

train_batches = tf_train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = tf_val_dataset.batch(BATCH_SIZE)
test_batches = tf_test_dataset.batch(BATCH_SIZE)


# ## 3. ResNet 모델 로드 

# #### 3.1 ResNet 모델 설정 및 로드

# In[ ]:


get_ipython().system('pip install keras_resnet')


# In[ ]:


import tensorflow as tf
import keras
import keras_resnet.models
from tensorflow.keras import layers

tf.random.set_seed(42)


# In[ ]:


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
CLASSES = 2


# In[ ]:


# model 선언 및 weight 초기화

x = keras.layers.Input(IMG_SHAPE)
model = keras_resnet.models.ResNet18(x, classes=CLASSES)

for layer in model.layers:
    try:
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)  #  tf.keras.initializers.he_normal
        layer.kernel_initializer = initializer
    except:
        pass


# In[ ]:


# tensorflow 프레임워크에서 제공하는 ResNet50을 활용하되 fully connected layer는 제외
# model = tf.keras.applications.resnet50.ResNet50(
#         input_shape=IMG_SHAPE, include_top=False, weights=None)

x = keras.layers.Input(IMG_SHAPE)
model = keras_resnet.models.ResNet18(x, classes=CLASSES)
model.compile("adam", "categorical_crossentropy", ["accuracy"])


#  ![03_resnet](04_02_file/03_resnet.png)

# In[ ]:


model.summary()


# In[ ]:


for image_batch, label_batch in train_batches.take(1):
    pass

print('image_batch: ', image_batch.shape)
print('label_batch: ', label_batch.shape)


# In[ ]:


out_sample = model(image_batch)


# print('out_sample: ', out_sample.shape)
# prediction_layer = keras.layers.Dense(1) # Fully connected Layer 추가
# prediction_batch = prediction_layer(out_sample) # (Batch, 2048) -> # (Batch, 1)

print("모델의 층 확인: ", len(model.layers))
print('입력 데이터 형태 :', image_batch.shape)
print('출력 데이터 형태 :', out_sample.shape)
print('label 데이터 형태: ', label_batch.shape)


# #### 3.2 ResNet Pooling Layer 및 Classification Layer 추가 
# 
# ---
# 
#  ![03_average_pool](04_02_file/03_average_pool.png)

# In[ ]:


out_sample = model(image_batch)
print('입력 데이터 형태 :', image_batch.shape)
print('출력 데이터 형태 :', out_sample.shape)


# ## 4. 하이퍼 파라미터 설정 및 컴파일

# #### 4.1 하이퍼 파라미터 설정
# ---
# 모델학습에 영향을 줄 수 있는 하이퍼 파라미터를 설정합니다.

# In[ ]:


EPOCH_NUM = 10
LEARNING_RATE = 0.001
# OPTIMIZER = 'RMSprop'
# OPTIMIZER = 'SGD'
OPTIMIZER = 'Adam'
PATIENCE = 10
MOMENTUM = 0.9


# #### 4.2  Optimizer 를 로드
# ---
# 딥러닝 모델 최적화를 위한 Optimizer 를 설정합니다.
# 
# Optimizer 설정 시 학습률 스케줄링을 적용합니다.
# 
#  ![04_lr_schedule](04_02_file/04_lr_schedule.png)
# 

# In[ ]:


# 딥러닝 모델을 위한 라이브러리
import keras # 케라스 라이브러리
from tensorflow.keras.optimizers import RMSprop, Adam, Adagrad, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay


# In[ ]:


lr_schedule = ExponentialDecay(LEARNING_RATE, decay_steps=50, decay_rate=0.9, staircase=True)


# In[ ]:


# Optimizer 결정
if OPTIMIZER == 'RMSprop':
    optimizer = RMSprop(learning_rate=lr_schedule)
elif OPTIMIZER == 'Adam':
    optimizer = Adam(learning_rate=lr_schedule)
elif OPTIMIZER == 'Adagrad':
    optimizer = Adagrad(learning_rate=lr_schedule)
elif OPTIMIZER =='SGD':
    optimizer = SGD(learning_rate=lr_schedule)
else:
    raise NotImplementedError


# #### 4.3 모델 컴파일
# ---
# 학습 모델에 Optimizer와 Loss 함수를 붙여 학습을 합니다.

# In[ ]:


model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# ## 5. ResNet 모델 학습 수행: ResNet 모델을 통한 학습 수행, 평가 및 예측 수행
# 

# #### 5.1 Callback 함수 설정
# ---
# * 과적합을 방지하기 위해 Validation loss가 증가하는 시점에서 학습을 중단합니다 (EarlyStopping).

# In[ ]:


from keras.callbacks import EarlyStopping
callbacks = []
if PATIENCE > 0:
    callbacks.append(EarlyStopping(monitor='val_loss', mode='min', patience=PATIENCE))


# In[ ]:


tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs")
callbacks.append(tensorboard)


# In[ ]:


checkpoint_path = "resnet_training/cp-{epoch:02d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 모델의 가중치를 저장하는 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

callbacks.append(cp_callback)


# #### 5.2 ResNet 모델 학습

# In[ ]:


history = model.fit(train_batches,
                         epochs=EPOCH_NUM,
                         validation_data=validation_batches,
                         callbacks=callbacks)


# #### 5.3 ResNet 모델 테스트

# In[ ]:


import pandas as pd

# 학습된 model weight 불러오기
model.load_weights('resnet_training/cp-01.ckpt')

loss, accuracy = model.evaluate(test_batches)
print('Test loss :', loss)
print('Test accuracy :', accuracy)


# ## 6. 제출

# ※ 과제 제출 관련하여 jupyter notebook 최상단의 `randomseed(=42)`를 절대 수정하시 마세요
# 
# ---
# 
# Renset 모델을 구현하여 테스트 데이터(**test_batches**)를 추론해보세요.
# 
# 목표 성능을 달성하기 위해서 전처리(`IMG_SIZE`, `SCALING`) 조정, 모델 수정(`KERNEL_SIZE`, `CHANNEL_SIZE`, `FC_SIZE`), 학습 하이퍼 파라미터(`EPOCH_NUM`, `BATCH_SIZE`, `LEARNING_RATE`, `OPTIMIZER`) 조정, 과적합 억제 및 초기화 기법(`Dropout`, `Weight decay`, `Weight Initialization`) 활용 등 어떤 방법을 활용하여도 좋습니다.
# 
# 추론 결과를 아래 표와 같은 포맷의 csv 파일로 저장해주세요.
# 
# |  | label |
# |-------|------|
# | 0     | 0 |
# | 1     | 1 |
# | 2     | 1 |
# | 3     | 0 |
# | 4     | 0 |
# 
# 위처럼, 테스트 데이터(**test_batches**)와 같은 순서로 정렬된 `index`와 그에 대한 `label`를 열로 갖는 dataframe을 `submission.csv` 로 저장합니다.
# 
# 
# `accuracy` 측정 결과 정확도가 **0.55 이상**이면 100점입니다.
# 
# (부분점수 있음)

# ### 채점

# In[ ]:


import pandas as pd

# 학습된 model weight 불러오기
model.load_weights('resnet_training/cp-01.ckpt')

prediction = model.predict(test_batches)
prediction = np.argmax(prediction, axis=1)

# 제출할 dataframe을 아래 코드에 대입하여 submission.csv 파일로 저장합니다.

answer_df = pd.DataFrame(prediction)
answer_df.columns = ['label']

answer_df.to_csv('submission.csv', index=False)


# 결과 csv 파일을 저장 후, 아래 코드를 실행하면 채점을 받을 수 있습니다.
# 
# **아래 코드를 수정하면 채점이 불가능 합니다.**

# In[ ]:


# 채점을 수행하기 위하여 로그인
import sys
sys.path.append('vendor')
from elice_challenge import check_score, upload


# In[ ]:


# 제출 파일 업로드
await upload()


# In[ ]:


# 채점 수행
await check_score()

