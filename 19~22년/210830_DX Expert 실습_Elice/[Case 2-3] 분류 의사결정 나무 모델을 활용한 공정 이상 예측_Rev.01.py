#!/usr/bin/env python
# coding: utf-8

# # [Case2-3] 반도체 공정 데이터를 활용한 공정 이상 예측_Rev.01

# ---

# 
# ## 프로젝트 목표
# ---
# - 의사결정나무 기반 이진 분류 모델 구현.
# - 결측값 처리, 특질 선택, 특질 추출 등 정형 데이터 전처리 작업.
# - 학습 데이터 불균형 문제 완화 기법.
# - 의사결정나무 분류 모델 하이퍼파라미터 튜닝 기법.

# ## 프로젝트 목차
# ---
# 
# 1. **정형 데이터 읽기:** Local에 저장되어 있는 정형 데이터를 불러오고 확인
# 
# 2. **데이터 전처리:** 모델에 필요한 입력 형태로 데이터 처리
# 
# 3. **의사 결정 나무 모델 정의 :** 의사 결정 나무 모델 구현
# 
# 4. **하이퍼 파라미터 설정 및 컴파일 :** 올바른 하이퍼 파라미터 설정
# 
# 5. **모델 학습 수행:** <span style="color:red"> 의사 결정 나무  </span>을 통한 학습 수행, 평가 및 예측 수행
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
# |Run|Epochs|Scaler|criterion|학습n_trees|max_depth|n_batcheslit|min_leaf|criterion|샘플링학습n_trees|max_depth|n_batches|epoch|변수|lr|MSE|MAE|
# |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# |01|-|Standard||1|2|1|||1|2|1|1|전부|-|0.965289|0.000000|
# |02|-|MinMax||1|2|1|||1|2|1|1|전부|-|0.864662|0.234043|
# |03|-|MinMax||1|4|1|||1|4|1|1|전부|-|0.856604|0.208333|
# |04|-|MinMax||1|2|1|||1|2|1|10|전부|-|에러|에러|
# |05|-|Standard||100|3|2|||1|2|1|1|전부|-|||
# 
# class 0 / class 1의 f1-score가 각각 0.90 / 0.23 이상이면 100점
# 
# 목표 성능을 달성하기 위해서 아래 어떤 방법을 활용하여도 좋습니다.
# 
# - 데이터 스케일링
# - upsampling 방법 조정 등
# 
# 
# - 'n_trees': 10,
# - 'max_depth': 5,
# - 'n_batches_per_layer': 1,
# - 'learning_rate': 0.005,
# 
# 피쳐 스케일링 안하기
# max_depth 2~3으로 조정하기
# 
# - 'n_trees': 100,
# - 'max_depth': 3,
# - 'n_batches_per_layer': 2

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


df = df.fillna(df.mean())


# In[ ]:


df


# ## 3. 데이터 시각화

# ### 3.1 데이터 분석
# 
# - 데이터의 통계치 (개수, 평균, 분산, 표준편차) 확인. 

# In[ ]:


df.describe()


# - 데이터의 상관 관계 분석

# In[ ]:


plt.figure(figsize=(30, 25))
p = sns.heatmap(df.corr()) # 데이터 컬럼 간, 상관관계 시각화. 하얀색이 더 높은 상관관계.


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

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


# In[ ]:


# sklearn 라이브러리 사용시 column name이 자동 indexing 되며 tf 에러가 발생해 스케일러 직접 구현

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# st_scaler = StandardScaler()

# normed_train_data = st_scaler.fit_transform(train_dataset)
# normed_test_data = st_scaler.fit_transform(test_dataset)
# normed_train_data = pd.DataFrame(normed_train_data)
# normed_test_data = pd.DataFrame(normed_test_data)


# In[ ]:


normed_train_data.head() # 스케일링 된 값 확인.


# In[ ]:


# 결측치 값 제거.
normed_train_data = normed_train_data.dropna(axis=1)
normed_test_data = normed_test_data.dropna(axis=1)


# ### 4.3 클래스 불균형 완화 기법
# 
# - 오버 샘플링 : 학습 과정 동안 데이터가 적은 클래스에서 의도적으로 더 자주 표본을 추출하여 학습 데이터를 구성.
# - 언더 샘플링 : 학습 과정 동안 데이터가 많은 클래스에서 표본을 의도적으로 더 적게 추출하여 학습 데이터를 구성.

# In[ ]:


get_ipython().system('pip install imblearn')


# In[ ]:


from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE


# In[ ]:


ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(normed_train_data, train_labels)


# In[ ]:


smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_resample(normed_train_data, train_labels)


# In[ ]:


cnt_zero = 0
cnt_one = 0
for l in y_resampled.values:
    if l == 1:
        cnt_one+=1
    else:
        cnt_zero+=1

print(f"label ratio -> {cnt_zero} : {cnt_one}")


# ## 5. 머신러닝 모델

# ### 5.3 의사결정 나무 
# ---
# 
# - 트리의 루트부터 시작해서 불순도가 가장 작아지는 특성으로 데이터를 분할하는 알고리즘.
# - 리프노드의 불순도가 0이 될 때까지 분할 작업 진행.
# - 분류결과가 도출되는 과정을 설명 가능함.

#  ![05_decision_tree](02_03_file/05_decision_tree.png)

# In[ ]:


import tensorflow as tf
from tensorflow import keras    
from tensorflow.keras import layers


# In[ ]:


def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(len(y))
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(len(y))
        return dataset
    return input_fn


# In[ ]:


normed_train_data


# In[ ]:


feature_columns = []
for feature_name in normed_train_data.keys():
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


# #### 5.3.1 학습

# In[ ]:


params = {
  'n_trees': 100,
  'max_depth': 3,
  'n_batches_per_layer': 2,
  # DFC를 가져오려면 center_bias = True로 설정해서
  # 모델이 피쳐를 적용하기 전에 초기 예측을 하도록 합니다
  'center_bias': True
}

train_input_fn = make_input_fn(normed_train_data, train_labels)

est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
est.train(train_input_fn, max_steps=100)


# #### 5.3.2 예측 및 평가
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

# In[ ]:


from IPython.display import clear_output
from sklearn.metrics import roc_curve

eval_input_fn = make_input_fn(normed_test_data, test_labels, shuffle=False, n_epochs=1)
result = est.evaluate(eval_input_fn)
clear_output()

print(pd.Series(result))


# In[ ]:


import sklearn.metrics as metrics
import matplotlib.pyplot as plt


# In[ ]:


preds = est.predict(eval_input_fn)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

pred_class = []
for pred in preds:
    pred_class.append(pred['class_ids'])

print(classification_report(test_labels, pred_class ))


# In[ ]:


preds = est.predict(eval_input_fn)

pred_score=[]
for pred in preds:
    pred_score.append(pred['probabilities'])


# In[ ]:


probs = np.array(pred_score)

preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(test_labels, preds)
print(fpr.shape, tpr.shape)

roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# #### 5.3.3 샘플링 데이터로 학습

# In[ ]:


feature_columns = []
for feature_name in X_resampled.keys():
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


# In[ ]:


params = {
  'n_trees': 1,
  'max_depth': 2,
  'n_batches_per_layer': 1,
  # DFC를 가져오려면 center_bias = True로 설정해서
  # 모델이 피쳐를 적용하기 전에 초기 예측을 하도록 합니다
  'center_bias': True
}

train_input_fn = make_input_fn(X_resampled, y_resampled)

est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
est.train(train_input_fn, max_steps=200)


# In[ ]:


from IPython.display import clear_output
from sklearn.metrics import roc_curve

eval_input_fn = make_input_fn(normed_test_data, test_labels, shuffle=False, n_epochs=1)
result = est.evaluate(eval_input_fn)
clear_output()

# 샘플링 기법 후, precision 및 recall 상승.
print(pd.Series(result))


# In[ ]:


preds = est.predict(eval_input_fn)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

pred_class = []
for pred in preds:
    pred_class.append(pred['class_ids'])

print(classification_report(test_labels, pred_class ))


# In[ ]:


preds = est.predict(eval_input_fn)

pred_score=[]
for pred in preds:
    pred_score.append(pred['probabilities'])


# In[ ]:


probs = np.array(pred_score)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(test_labels, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


pred_dicts = list(est.experimental_predict_with_explanations(eval_input_fn))


# In[ ]:


labels = test_labels.values
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])
df_dfc.describe().T


# In[ ]:


# 시각화의 표준 양식입니다.
def _get_color(value):
    """양의 DFC를 초록색으로 음의 DFC를 빨간색으로 표시합니다."""
    green, red = sns.color_palette()[2:4]
    if value >= 0: return green
    return red

def _add_feature_values(feature_values, ax):
    """피쳐 값을 플롯의 왼쪽에 배치합니다."""
    x_coord = ax.get_xlim()[0]
    OFFSET = 0.15
    for y_coord, (feat_name, feat_val) in enumerate(feature_values.items()):
        t = plt.text(x_coord, y_coord - OFFSET, '{}'.format(feat_val), size=12)
        t.set_bbox(dict(facecolor='white', alpha=0.5))
    from matplotlib.font_manager import FontProperties
    font = FontProperties()
    font.set_weight('bold')
    t = plt.text(x_coord, y_coord + 1 - OFFSET, 'feature\nvalue',
    fontproperties=font, size=12)

def plot_example(example):
    TOP_N = 8 # 위에서부터 8개의 피쳐를 봅니다.
    sorted_ix = example.abs().sort_values()[-TOP_N:].index  # 중요도를 정렬합니다.
    example = example[sorted_ix]
    ax = example.to_frame().plot(kind='barh',
                          legend=None,
                          alpha=0.75,
                          figsize=(10,6))
    ax.grid(False, axis='y')
    ax.set_yticklabels(ax.get_yticklabels(), size=14)

    # 피쳐 값들을 넣습니다.
    _add_feature_values(normed_test_data.iloc[ID][sorted_ix], ax)
    return ax


# In[ ]:


# 시각화 결과입니다.
ID = 17
example = df_dfc.iloc[ID]  # 검증 데이터셋에서 i번째 데이터를 선택합니다.
TOP_N = 8  # 위에서 부터 8개의 피쳐를 확인합니다.
sorted_ix = example.abs().sort_values()[-TOP_N:].index
ax = plot_example(example)
ax.set_title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))
ax.set_xlabel('Contribution to predicted probability', size=14)
plt.show()


# ## 6. 제출

# ※ 과제 제출 관련하여 jupyter notebook 최상단의 `randomseed(=42)`를 절대 수정하시 마세요
# 
# ---
# 
# 의사결정나무 분류 모델을 구현하여 표준 스케일링 된 테스트 데이터(**normed_test_data**)를 추론해보세요.
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
# `sklearn.metrics`의 `classification_report`을 통한 성능 측정으로 **class 0 / class 1의 f1-score**가 각각 **0.90 / 0.23** 이상이면 100점입니다.
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

answer_df = pd.DataFrame(pred_class)
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

