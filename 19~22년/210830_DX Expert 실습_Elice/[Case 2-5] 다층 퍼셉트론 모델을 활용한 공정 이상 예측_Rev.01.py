#!/usr/bin/env python
# coding: utf-8

# # [Case2-5] 반도체 공정 데이터를 활용한 공정 이상 예측_Rev.01

# ---

# 
# ## 프로젝트 목표
# ---
# - 다층퍼셉트론 기반 이진 분류 모델 구현.
# - 결측값 처리, 특질 선택, 특질 추출 등 정형 데이터 전처리 작업.
# - 학습 데이터 불균형 문제 완화 기법.
# - 다층퍼셉트론 분류 모델 하이퍼파라미터 튜닝 기법.

# ## 프로젝트 목차
# ---
# 
# 1. **정형 데이터 읽기:** Local에 저장되어 있는 정형 데이터를 불러오고 확인
# 
# 2. **데이터 전처리:** 모델에 필요한 입력 형태로 데이터 처리
# 
# 3. **다층 퍼셉트론 모델 정의 :** 다층 퍼셉트론 모델 구현
# 
# 4. **하이퍼 파라미터 설정 및 컴파일 :** 올바른 하이퍼 파라미터 설정
# 
# 5. **모델 학습 수행:** <span style="color:red"> 다층 퍼셉트론 </span>을 통한 학습 수행, 평가 및 예측 수행
# 
# 6. **제출:** 예측한 결과를 제출한 후 채점 결과를 확인합니다.

# ## 데이터 출처
# ---
# https://www.kaggle.com/paresh2047/uci-semcom

# ## 프로젝트 개요
# ---
# 
# **데이터:** 센서 및 측정으로 수집된 반도체 제조 공정 정보와 그에 따른 공정 이상 여부 데이터.
# 
# **가정:** 측정된 제조 공정 정보에 결함 여부를 판단할 수 있는 특질이 존재.
# 
# **목표:** 결함이 있는 반도체를 출고전에 제외.
# 
# **설명:** 
# 
# ```
# 데이터에는 결측값이 존재.
# 측정된 시그널(특질)들은 매우 다양하지만 (590개) 불필요한 정보나 노이즈를 포함.
# 소량의 학습 데이터 제공(총 1567개).
# 클래스간 학습 데이터 수 불균형함 (이상없음: 1463개, 이상:104개)
# ```

# ## 결과 요약
# 
# 각 파라미터 조건에 따른 모델 평가 결과는 아래와 같다.
# 
# |Run|Epochs|Scaler|criterion|frac|max_iter|preds>|Dense|criterion|n_trees|max_depth|n_batches|epoch|변수|lr|class0|class1|
# |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# |01|-|Standard||||0.5|30|||||||0.001|0.938356|0.142857|
# |02|-|Robust||||0.5|30|||||||0.001|0.939130|0.313725|
# |03|-|Robust||||0.5|30|||||||0.0001|0.908766|0.238806|
# |04|-|Robust||||0.5|30|||||||0.01|0.943201|0.266667|
# |05|-|Robust||||0.8|30|||||||0.001|0.947368|0.162162|
# |06|-|Robust||||0.5|100|||||||0.001|0.944828|0.304348|
# 
# class 0 / class 1의 f1-score가 각각 0.93 / 0.34 이상이면 100점
# 
# 목표 성능을 달성하기 위해서 아래 어떤 방법을 활용하여도 좋습니다.
# 
# - 데이터 스케일링
# - upsampling 방법 조정 등

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
# 데이터 프레임 형태의 데이터를 pandas 라이브러리를 이용해 불러오자.

# In[ ]:


df = pd.read_csv("secom.data", sep=" ", names = [str(i)+"th col" for i in range(590)])


# In[ ]:


df.head()


# In[ ]:


df.shape # 데이터 형태 확인.


# ## 2. 데이터 정제
# ---

# ### 2.1 데이터 결측치 보간

# In[ ]:


df = df.drop([], axis=1)


# In[ ]:


df = df.fillna(df.mean())


# In[ ]:


df


# ## 3. 데이터 시각화

# ### 3.1 데이터 정제
# 
# - 데이터의 통계치 (개수, 평균, 분산, 표준편차) 확인. 

# In[ ]:


df.describe()


# - 데이터의 상관 관계 분석

# ## 4. 데이터 전처리

# In[ ]:


labels = pd.read_csv("secom_labels.data", sep=" ", names = ['label', 'date'])


# In[ ]:


labels = labels.drop(['date'], axis=1)
labels = labels.replace(-1, 0)


# In[ ]:


labels


# ### 4.1 학습/평가 데이터셋 분리

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_dataset = df.sample(frac=0.8,random_state=0)
test_dataset = df.drop(train_dataset.index)
train_labels = labels.loc[train_dataset.index]
test_labels = labels.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()


# In[ ]:


print("train data shape : ", train_dataset.shape)
print("test data shape : ", test_dataset.shape)


# In[ ]:


train_stats


# ### 4.2 특성 스케일링

# In[ ]:


from sklearn.preprocessing import StandardScaler # 표준 스케일링  (평균 = 0 / 표준편차 = 1)
from sklearn.preprocessing import MinMaxScaler  # 최대/최소 스케일링 (이상치에 취약)
from sklearn.preprocessing import RobustScaler  # 중앙값 = 0 / IQR(1분위(25%) ~ 3분위(75%)) = 1 (이상치 영향 최소화, 넓게 분포)
from sklearn.preprocessing import MaxAbsScaler  # |x|  <= 1 , 이상치에 취약할 수 있다. 양수만 있는 데이터의 경우 MinMaxScaler 유사


# In[ ]:


# st_scaler = StandardScaler()
st_scaler = RobustScaler()

normed_train_data = st_scaler.fit_transform(train_dataset)
normed_test_data = st_scaler.fit_transform(test_dataset)
normed_train_data = pd.DataFrame(normed_train_data)
normed_test_data = pd.DataFrame(normed_test_data)


# In[ ]:


normed_train_data.head() # 스케일링 된 값 확인.


# In[ ]:


# 결측치 값 제거.
normed_train_data = normed_train_data.dropna(axis=1)
normed_test_data = normed_test_data.dropna(axis=1)


# In[ ]:


# 데이터 특성값 범위 확인

import matplotlib.pyplot as plt
import matplotlib


# matplotlib 설정
matplotlib.rc('font', family='AppleGothic') # 한글 설정
plt.rcParams['axes.unicode_minus'] = False # -표시

# feature visualization
plt.boxplot(normed_test_data[50], manage_ticks=False) # 데이터, 소눈금 표시 안하기
plt.yscale('symlog') # 축 스케일을 log 로
plt.xlabel('feature list') # x축 이름
plt.ylabel('feature') # y축 이름
plt.show() # 그래프 출력


# ### 4.3 클래스 불균형 완화 기법
# 
# - 오버 샘플링 : 학습 과정 동안 데이터가 적은 클래스에서 의도적으로 더 자주 표본을 추출하여 학습 데이터를 구성.
# - 언더 샘플링 : 학습 과정 동안 데이터가 많은 클래스에서 표본을 의도적으로 더 적게 추출하여 학습 데이터를 구성.

# In[ ]:


get_ipython().system('pip install imblearn')


# In[ ]:


from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler


# In[ ]:


ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(normed_train_data, train_labels)


# In[ ]:


cnt_zero = 0
cnt_one = 0
for l in y_resampled.values:
    if l == 1:
        cnt_one+=1
    else:
        cnt_zero+=1

print(f"label ratio -> {cnt_zero} : {cnt_one}")


# ## 5. 딥뉴럴 네트워크 모델

# ### 5.5 다층 퍼셉트론 모델 구현
# 
# - 로지스틱 회귀와는 다르게 인풋과 아웃풋 사이의 숨겨진(hidden) 비선형 레이어들을 하나 이상 포함하여 m차원의 데이터를 o 원하는 수의 아웃풋으로 결과를 내도록 학습. 
# - 이진 분류의 경우 0과 1 사이로 결과가 나올 수 있도록 학습.
# - 매우 복잡한 비선형 모델도 표현가능하므로 복잡한 데이터 분포도 학습 가능함.

# In[ ]:


import tensorflow as tf
from tensorflow import keras    
from tensorflow.keras import layers


# In[ ]:


# 하이퍼 파라미터 정의
lr = 0.001
training_epoch = 10

# 평가 메트릭 정의.
auc = tf.keras.metrics.AUC(num_thresholds=3)
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
tp = tf.keras.metrics.TruePositives()
fn = tf.keras.metrics.FalseNegatives()


# In[ ]:


def logistic_regression():
    model = keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=[normed_train_data.shape[1]]) # 결측 열 제거.
  ])
    
    
    optimizer = tf.keras.optimizers.RMSprop(lr)
    loss_object = tf.keras.losses.BinaryCrossentropy()
    
    model.compile(loss=loss_object,
                optimizer=optimizer,
                metrics=['accuracy', auc, precision, recall, tp, fn])
    return model

logi_model = logistic_regression()


# In[ ]:


def multi_layer_perceptron():
    model = keras.Sequential([
    layers.Dense(100, activation='relu', input_shape=[normed_train_data.shape[1]]), # len(train_dataset.keys())결측 열 제거.
    # layers.Dense(100, activation='relu'),
    layers.Dense(1, activation='sigmoid')
  ])
    
    optimizer = tf.keras.optimizers.RMSprop(lr)
    loss_object = tf.keras.losses.BinaryCrossentropy()
    
    model.compile(loss=loss_object,
                optimizer=optimizer,
                metrics=['accuracy', auc, precision, recall, tp, fn])
    return model

mlp_model = multi_layer_perceptron()


# In[ ]:


mlp_model.summary()


# #### 5.5.1 학습

# In[ ]:


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)


# In[ ]:


mlp_model.fit(normed_train_data, train_labels,
              validation_split = 0.2, epochs=training_epoch, verbose=2, callbacks=[early_stop])


# #### 5.5.2 예측 및 평가
# 
# 불균형한 데이터의 분류 성능을 평가하기 위해, Accuracy 이외에 정밀도, 재현율, 이 둘의 조화평균인 F1 score를 구함.
# 
# - precision : 모델이 positive라고 분류한 것들 중, 진짜 positive 데이터 비율. tp / tp + fp
# - recall : 모델이 맞게 분류한 것들 중, positive를 데이터를 맞춘 비율. tp / tp + fn
# - f1 score : 2 * precision * recall / precision + recall 
# 
# 클래스가 불균형할 수록 정확도 이외의 위와같은 지표들을 같이 살펴보아야 분류기 성능을 비교하여 측정 가능.
# 
# - roc curve : x축을 False Positive Rate y축을 Recall(True Positive Rate)로 두고 시각화한 그래프.
# - roc curve의 아래면적을 AUC로 커브의 꼭지가 1에 가까울수록 분류기의 성능이 좋음을 의미.
# 
# - BER : 각 클래스의 오류율의 평균으로 0일때 완벽한 분류기, 0.5일때 랜덤 선택 수준의 분류기.
#     - 1/2(𝐹𝑁𝑅+𝐹𝑃𝑅)

# In[ ]:


scores = mlp_model.evaluate(normed_test_data, test_labels, verbose=2)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

predict = mlp_model.predict(normed_test_data)
predict = np.where(predict > 0.5, 1, 0)
print(classification_report(predict, test_labels))


# In[ ]:


import sklearn.metrics as metrics
import matplotlib.pyplot as plt


# In[ ]:


# preds = mlp_model.predict_proba(normed_test_data)
preds = mlp_model.predict(normed_test_data)


# In[ ]:


fpr, tpr, threshold = metrics.roc_curve(test_labels, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# #### 5.5.3 샘플링 데이터 학습
# 

# In[ ]:


mlp_model = multi_layer_perceptron()


# In[ ]:


mlp_model.fit(X_resampled, y_resampled,
              validation_split = 0.2, epochs=training_epoch, verbose=2, callbacks=[early_stop])


# In[ ]:


scores = mlp_model.evaluate(normed_test_data, test_labels, verbose=2)


# In[ ]:


preds = mlp_model.predict(normed_test_data)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

preds = np.where(preds > 0.5, 1, 0)
print(classification_report(preds, test_labels))

res = classification_report(preds, test_labels, output_dict=True)


# In[ ]:


fpr, tpr, threshold = metrics.roc_curve(test_labels, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# #### 5.5.3 모델 앙상블

# In[ ]:


num_ensemble_models = 2
training_epoch = 3

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# 모델을 여러개를 생성해서 모델 리스트를 만듦.
models = [] 
for m in range(num_ensemble_models):
    models.append(multi_layer_perceptron())


# In[ ]:


# 모델리스트에서 정의된 모델을 하나씩 가져와서 학습을 진행함.
for i, model in enumerate(models):
    model.fit(normed_train_data, train_labels,
              validation_split = 0.2, epochs=training_epoch, verbose=2, callbacks=[early_stop])
    print("model %d completion!" % i)
    # 학습이 끝난 모델을 저장함.
    model.save_weights("my_checkpoint"+str(i))
    


# In[ ]:


pred_list = []
for i, model in enumerate(models):
    pred = model.predict(normed_test_data)
    pred_list.append(pred)
    
predictions = sum(pred_list)/len(pred_list)
predictions = np.where(predictions > 0.5, 1, 0)

print(classification_report(predictions, test_labels))


# #### 5.5.3 customized thresholding

# In[ ]:


pred_list


# In[ ]:


pred_list = []
for i, model in enumerate(models):
    pred = model.predict(normed_test_data)
    pred_list.append(pred)
    
predictions = sum(pred_list)/len(pred_list)
predictions = np.where(predictions > 0.5, 1, 0)

print(classification_report(predictions, test_labels))


# ## 6. 제출

# ※ 과제 제출 관련하여 jupyter notebook 최상단의 `randomseed(=42)`를 절대 수정하시 마세요
# 
# ---
# 
# 다층퍼셉트론 분류 모델을 구현하여 표준 스케일링 된 테스트 데이터(**normed_test_data**)를 추론해보세요.
# 
# 목표 성능을 달성하기 위해서 `데이터 스케일링`, `upsampling 방법` 조정 등 어떤 방법을 활용하여도 좋습니다.
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
# 위처럼, 테스트 데이터(**normed_test_data**)와 같은 순서로 정렬된 `index`와 그에 대한 `label`을 열로 갖는 dataframe을 `submission.csv` 로 저장합니다.
# 
# `sklearn.metrics`의 `classification_report`을 통한 성능 측정으로 **class 0 / class 1의 f1-score**가 각각 **0.93 / 0.34** 이상이면 100점입니다.
# 
# (부분점수 있음)
# 
# 

# ### 채점

# 결과 csv 파일을 저장 후, 아래 코드를 실행하면 채점을 받을 수 있습니다.
# 
# **아래 코드를 수정하면 채점이 불가능 합니다.**

# In[ ]:


# 제출할 dataframe을 아래 코드에 대입하여 submission.csv 파일로 저장합니다.

answer_df = pd.DataFrame(preds)
answer_df.columns = ['label']

print(answer_df)
answer_df.to_csv('submission.csv', index=False)


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

