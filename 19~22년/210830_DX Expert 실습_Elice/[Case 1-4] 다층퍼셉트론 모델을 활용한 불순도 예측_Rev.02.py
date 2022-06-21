#!/usr/bin/env python
# coding: utf-8

# # [Case1-4] 광석 처리 공단에서의 이산화규소 (silica) 불순도 예측_Rev.02

# ---

# 
# ## 프로젝트 목표
# ---
# - **다층퍼셉트론 회귀 분석 모델을 구현**.
# - 불필요한 입력값을 제거하는 데이터 정제 기법.
# - 데이터 스케일링 작업.
# - 학습, 평가, 검증 데이터 전처리 작업.
# - 다층퍼셉트론 하이퍼파라미터 튜닝 기법.
# - **다층퍼셉트론 회귀 모델의 과적합 억제 기법**.
# 

# ## 프로젝트 목차
# ---
# 
# 1. **정형 데이터 읽기:** Local에 저장되어 있는 Kaggle 데이터를 불러오고 데이터 프레임 확인
# 
# 2. **데이터 전처리:** 모델에 필요한 입력 형태로 데이터 처리
# 
# 3. **다층 퍼셉트론 모델 정의 :** 다층 퍼셉트론 모델 구현
# 
# 4. **하이퍼 파라미터 설정 및 컴파일 :** 올바른 하이퍼 파라미터 설정
# 
# 5. **모델 학습 수행:** <span style="color:red">다층 퍼셉트론</span>을 통한 학습 수행, 평가 및 예측 수행
# 
# 6. **제출:** 예측한 결과를 제출한 후 채점 결과를 확인합니다.
# 

# 각 파라미터 조건에 따른 모델 평가 결과는 아래와 같다.
# 
# |Run|앙상블 모델수|hidden_size|hiddens|max_iter|power_t|min_split|min_leaf|max_depth_앙상블|epochs|lr_scheduling|lr|MSE|MAE|
# |-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# |01|2|32|32*4|-|-|-|-|-|2|adaptive|0.00001|1.043052|0.792862|
# |02|2|32|32*4|-|-|-|-|-|10|adaptive|0.00001|0.888511|0.729321|
# 
# mse < 0.45, mae < 0.49가 100점

# - 목표 성능을 달성하기 위해서
#     - 앙상블 모델 수(num_ensemble_models) 조정,
#     - 은닉층의 크기(hidden_size) 조정,
#     - 학습 하이퍼 파라미터
#         - learning_rate
#         - activation_function
#         - epochs 조정 등

# ## 데이터 출처
# ---
# https://www.kaggle.com/edumagalhaes/quality-prediction-in-a-mining-process?select=MiningProcess_Flotation_Plant_Database.csv

# ## 프로젝트 개요
# ---
# 
# **데이터:** 실제 제조 공장, 특히 광산 공장의 데이터베이스로, 채굴 과정에서 가장 중요한 부분 중 하나 인 부유 플랜트 에서 나온 데이터.
# 
# **목표:** 주요 목표는이 데이터를 사용하여 광석 정광에 얼마나 많은 불순물이 있는지 예측.
# 
# **설명:** 
# 
# ```
# 1 번째 열: 시간 및 날짜 범위
# 2-3 번째 열: 부양 시설에 공급되기 직전에 철광석 펄프의 품질 측정.
# 4-8 번째 열: 공정이 끝날 때 광석 품질에 영향을 미치는 가장 중요한 변수.
# 9-22 번째 열: 공정 데이터 (광석 품질에도 영향을 미치는 부양 컬럼 내부의 레벨 및 공기 흐름)
# 23-24 번째 열: 실험실에서 예측한 최종 철광석 펄프 품질.
# 마지막 열을 정답(Target)으로 사용.
# ```

# 각 파라미터 조건에 따른 모델 평가 결과는 아래와 같다.
# 
# |Run|앙상블 모델수|hidden_size|hiddens|activation|변수|min_split|min_leaf|max_depth_앙상블|epochs|lr_scheduling|lr|MSE|MAE|
# |-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# |01|2|32|32*4|linear|전부|-|-|-|2|adaptive|0.00001|1.043052|0.792862|
# |02|2|32|32*4|linear|전부|-|-|-|10|adaptive|0.00001|0.888511|0.729321|
# |03|2|64|32*4|linear|전부|-|-|-|10|adaptive|0.00001|0.898572|0.735859|
# |04|4|32|32*4|linear|전부|-|-|-|10|adaptive|0.00001|0.888193|0.730502|
# |05|2|32|32*4|relu|전부|-|-|-|10|adaptive|0.00001|0.895494|0.735599|
# |06|2|32|256*2|relu|전부|-|-|-|10|adaptive|0.00001|0.642414|0.601313|
# |07|2|32|256*2|linear|전부|-|-|-|10|adaptive|0.00001|1.075808|0.817720|
# |08|2|32|256*2|linear|5개제거|-|-|-|10|adaptive|0.00001|1.135065|0.846097|
# |09|2|32|256*2|linear|5개제거|-|-|-|10|adaptive|0.001|1.137899|0.849548|
# 
# mse < 0.45, mae < 0.49가 100점
# 
# - 목표 성능을 달성하기 위해서
#     - 앙상블 모델 수(num_ensemble_models) 조정,
#     - 은닉층의 크기(hidden_size) 조정,
#     - 학습 하이퍼 파라미터
#         - learning_rate
#         - activation_function
#         - epochs 조정 등

# ## 1. 데이터 읽기 
# ---

# ### 1.1 라이브러리 불러오기

# In[ ]:


# 데이터 프레임 형태의 데이터 및 연산을 위한 라이브러리
import random
import numpy as np
import pandas as pd

# 시각화 및 학습 로그를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# 모델 재현성을 위한 랜덤시드 설정
random.seed(42)
np.random.seed(42)


# ### 1.2 데이터 불러오기
# ---
# 데이터 프레임 형태의 데이터를 pandas 라이브러리를 이용해 불러옵니다.

# In[ ]:


df = pd.read_csv("/mnt/data/chapter_1/MiningProcess_Flotation_Plant_Database.csv",decimal=",",parse_dates=["date"],infer_datetime_format=True,engine='python', error_bad_lines=False).drop_duplicates()


# In[ ]:


df_raw = df


# In[ ]:


df.head()


# In[ ]:


df.shape # 데이터 형태 확인.


# ## 2. 데이터 정제
# ---

# ### 2.1 데이터 결측치 제거

# In[ ]:


df = df.dropna() # 결측치 제거.
df.shape # 결측치가 없어 기존과 동일.


# ## 3. 데이터 시각화

# ### 3.1 데이터 정제
# 
# - 데이터의 통계치를 (개수, 평균, 분산, 표준편차) 확인합니다. 

# In[ ]:


df.describe()


# ### 3.2 데이터의 상관 관계 분석

# In[ ]:


SMALL_SIZE = 15
MEDIUM_SIZE = 20
WEIGHT = 'bold'
plt.rc('font', size=SMALL_SIZE, weight=WEIGHT) # controls default text sizes 
plt.rc('xtick', labelsize=MEDIUM_SIZE) # fontsize of the tick labels 
plt.rc('ytick', labelsize=MEDIUM_SIZE) # fontsize of the tick labels
plt.figure(figsize=(30, 25))

p = sns.heatmap(df.corr(), annot=True) # 데이터 컬럼 간, 상관관계 시각화. 하얀색이 더 높은 상관관계.


# ### 3.3 산점도 시각화

# In[ ]:


sns.set(rc = {'figure.figsize':(15,8)})

sns.pairplot(df[['% Silica Concentrate', 'Amina Flow', 'Ore Pulp pH', 'Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow', 'Flotation Column 03 Air Flow']], diag_kind='kde')


# ## 4. 데이터 전처리
# 
# - 모델 스스로 데이터의 특성을 학습하여 특별한 데이터의 정제나 특성 엔지니어링의 영향력이 크지 않음.

# In[ ]:


# 필요없다고 판단되는 피처 몇 가지를 제거.
df = df.drop(['% Iron Concentrate', 'Starch Flow', 'Amina Flow', 'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density'], axis=1)

# 'Starch Flow' 'Amina Flow' 'O're Pulp Flow' 'Ore Pulp pH' 'Ore Pulp Density'


# In[ ]:


df.head()


# In[ ]:


print(df.shape)


# ### 4.1 학습/평가 데이터셋 분리

# In[ ]:


train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index) 


# In[ ]:


train_stats = train_dataset.describe()
train_stats.pop("% Silica Concentrate")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('% Silica Concentrate')
test_labels = test_dataset.pop('% Silica Concentrate')

# 추후 결과 시각화를 위해 date column 확보
train_date = train_dataset.pop('date')
test_date = test_dataset.pop('date')


# ### 4.2 특성 스케일링

# In[ ]:


from sklearn.preprocessing import StandardScaler # 표준 스케일링  (평균 = 0 / 표준편차 = 1)
from sklearn.preprocessing import MinMaxScaler  # 최대/최소 스케일링 (이상치에 취약)
from sklearn.preprocessing import RobustScaler  # 중앙값 = 0 / IQR(1분위(25%) ~ 3분위(75%)) = 1 (이상치 영향 최소화, 넓게 분포)
from sklearn.preprocessing import MaxAbsScaler  # |x|  <= 1 , 이상치에 취약할 수 있다. 양수만 있는 데이터의 경우 MinMaxScaler 유사


# In[ ]:


st_scaler = StandardScaler()

normed_train_data = st_scaler.fit_transform(train_dataset)
normed_test_data = st_scaler.fit_transform(test_dataset)
normed_train_data = pd.DataFrame(normed_train_data)
normed_test_data = pd.DataFrame(normed_test_data)


# In[ ]:


normed_train_data.head() # 스케일링 된 값 확인.


# ## 5. 딥뉴럴 네트워크 모델

# ### 5.1 다층 퍼셉트론 모델 구현
# 
# - 매우 복잡한 비선형 모델도 표현가능하므로 복잡한 데이터 분포도 학습 가능함.

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:


# 하이퍼 파라미터 정의
lr = 0.001
act = 'linear'
training_epoch = 10
kernel_regularizer=keras.regularizers.l1(l=0.1)


# In[ ]:


# 단일 퍼셉트론 
def multiple_regression():
    model = keras.Sequential([
        layers.Dense(1, activation='relu', input_shape=[len(train_dataset.keys())], kernel_regularizer=keras.regularizers.l1(l=0.1))
    ])    
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

model = multiple_regression()


# In[ ]:


model.summary()


# In[ ]:


# 다층 퍼셉트론 모델 구현, 가운데에 있는 중간 사이즈는 자유롭게 지정.
def multi_layer_perceptron(hidden_size=32):
    model = keras.Sequential([
    layers.Dense(hidden_size, activation=act, input_shape=[len(train_dataset.keys())]),
    layers.Dense(hidden_size*2, activation=act, input_shape=[hidden_size]),
    layers.Dropout(0.1),
    layers.Dense(1)
  ])
    # 모델을 학습 시, Adam 옵티마이저 기법사용, 안정적인 최적화 달성.
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

model = multi_layer_perceptron()


# In[ ]:


model.summary()


# #### 5.1.1 학습

# In[ ]:


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs")
model.fit(
  normed_train_data, train_labels, 
  epochs=training_epoch, validation_split = 0.125, verbose=2, callbacks=[early_stop, tensorboard])

# 학습이 끝난 모델을 저장함.
model.save_weights(f"pt/my_checkpoint-{act}")


# #### 5.1.2 평가 및 예측

# In[ ]:


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)


# In[ ]:


logits = model.predict(normed_test_data)
print(f'prediction: {logits}')
print(f'labels: {test_labels}')


# In[ ]:


# target과 prediction 사이 비교 시각화

palette = ['#FB6542', '#FFBB00', '#3F681C', '#375E97']

flatten_logit = logits.flatten()


ci = 1.96*np.std(flatten_logit)/np.mean(flatten_logit)

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(18,6))
ax.plot(test_date, test_labels, color=palette[1], label='Actual Value')
ax.plot(test_date, flatten_logit, color=palette[2], label='Forecast')

ax.fill_between(test_date, (flatten_logit-ci), (flatten_logit+ci), color=palette[3],
                alpha=.1, label='95% Confidence Interval')
ax.set_title('%Silica in Concentrate: Actual Values and Forecasts by LSTM',
             loc='left', weight='bold', size=16)
ax.set_ylabel('Silica in Concentrate (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
plt.show()


# #### 5.1.3 앙상블 모델
# 
# - 여러 개의 모델을 조화롭게 학습시켜 그 모델들의 결과값을 이용함.
# - 각각 학습한 여러 모델들을 조합하여 일반화하므로 성능이 향상되는 효과.
# - 과적합 감소 효과도 존재.

# In[ ]:


num_ensemble_models = 2
training_epoch = 10

# 모델을 여러개를 생성해서 모델 리스트를 만듦.
models = [] 
hiddens = [256, 256]
for m in range(num_ensemble_models):
    models.append(multi_layer_perceptron(hidden_size=hiddens[m]))


# In[ ]:


# 모델을 학습할 때, 모델이 학습 데이터에 과적합 되지 않도록, 
# 적당한 시점에서 학습을 중단하는 Early Stopping을 사용.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3) # patience는 성능이 증가하지 않는 epoch을 몇 번 허용할 것인지 설정.
# 'logs' 디렉토리에다가 모델이 학습되는 것의 시각화 로그를 저장한다.
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs")

# 모델리스트에서 정의된 모델을 하나씩 가져와서 학습을 진행함.
for i, model in enumerate(models):
    model.fit(normed_train_data, train_labels, epochs=training_epoch, validation_split = 0.125, verbose=2, callbacks=[early_stop, tensorboard])
    print("model %d completion!" % i)
    # 학습이 끝난 모델을 저장함.
    # model.save_weights("my_checkpoint"+str(i)) # 네트워크 느려져서 주석 처리


# In[ ]:


# 각각의 모델들 (model1, model2, model3) 에 대해 테스트 데이터로 예측 값을 구하고, 
# 이 값들을 
logit_list = []
for i, model in enumerate(models):
    logits = model.predict(normed_test_data)
    logit_list.append(logits)

# 세 개의 예측 값들의 평균으로 구함.
predictions = sum(logit_list)/len(logit_list)
predictions = predictions.squeeze(-1)


# In[ ]:


print(f'predictions: {predictions}')
print(f'labels: {test_labels}')


# #### 평가

# In[ ]:


mse = tf.keras.losses.mean_squared_error(test_labels, predictions)
mae = tf.keras.losses.mean_absolute_error(test_labels, predictions)


# In[ ]:


print(f"Test mse: {mse.numpy():.4f}")
print(f"Test mae: {mae.numpy():.4f}")


# In[ ]:


# target과 prediction 사이 비교 시각화

palette = ['#FB6542', '#FFBB00', '#3F681C', '#375E97']

ci = 1.96*np.std(predictions)/np.mean(predictions)

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(18,6))
ax.plot(test_date, test_labels, color=palette[1], label='Actual Value')
ax.plot(test_date, predictions, color=palette[2], label='Forecast')

ax.fill_between(test_date, (predictions-ci), (predictions+ci), color=palette[3],
                alpha=.1, label='95% Confidence Interval')
ax.set_title('%Silica in Concentrate: Actual Values and Forecasts by LSTM',
             loc='left', weight='bold', size=16)
ax.set_ylabel('Silica in Concentrate (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
plt.show()


# ## 6. 제출

# ※ 과제 제출 관련하여 jupyter notebook 최상단의 `randomseed(=42)`를 절대 수정하시 마세요
# 
# ---
# 
# 다층 퍼셉트론 앙상블 모델을 구현하여 표준 스케일링 된 테스트 데이터(**normed_test_data**)를 추론해보세요.
# 
# 목표 성능을 달성하기 위해서 앙상블 모델 수(`num_ensemble_models`) 조정, 은닉층의 크기(`hidden_size`) 조정, 혹은 학습 하이퍼 파라미터(`learning_rate`, `activation_function`, `epochs`) 조정 등 어떤 방법을 활용하여도 좋습니다.
# 
# 
# 추론 결과를 아래 표와 같은 포맷의 csv 파일로 저장해주세요.
# 
# |  | silica |
# |-------|------|
# | 0     | 2.34 |
# | 1     | 1.26 |
# | 2     | 4.37 |
# | 3     | 0.30 |
# | 4     | 0.31 |
# 
# 위처럼, 테스트 데이터(**normed_test_data**)와 같은 순서로 정렬된 `index`와 그에 대한 `silica`를 열로 갖는 dataframe을 `submission.csv` 로 저장합니다.
# 
# 
# `sklearn.metrics`의 `mean_squared_error`, `mean_absolute_error`을 통한 성능 측정으로 채점되며 **mse < 0.45, mae < 0.49**가 100점입니다 
# 
# (부분 점수 있음).

# ### 채점

# 결과 csv 파일을 저장 후, 아래 코드를 실행하면 채점을 받을 수 있습니다.
# 
# **아래 코드를 수정하면 채점이 불가능 합니다.**

# In[ ]:


# 제출할 dataframe을 아래 코드에 대입하여 submission.csv 파일로 저장합니다.

answer_df = pd.DataFrame(predictions)
answer_df.columns = ['silica']

print(answer_df)
answer_df.to_csv('submission.csv',index=False)


# 위 코드 셀은 채점해야 하는 결과물의 형태에 따라 수정이 가능합니다. 
# 아래 3개의 셀은 수정하지 마세요. 

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

