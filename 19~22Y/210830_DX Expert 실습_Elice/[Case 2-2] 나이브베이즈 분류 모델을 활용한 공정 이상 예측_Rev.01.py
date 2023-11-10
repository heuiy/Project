#!/usr/bin/env python
# coding: utf-8

# # [Case2-2] 반도체 공정 데이터를 활용한 공정 이상 예측_Rev.01

# ---

# 
# ## 프로젝트 목표
# ---
# - 나이브베이즈 분류 모델 구현.
# - 결측값 처리, 특질 선택, 특질 추출 등 정형 데이터 전처리 작업.
# - 학습 데이터 불균형 문제 완화 기법.
# - 나이브베이즈 분류 모델 하이퍼파라미터 튜닝 기법.

# ## 프로젝트 목차
# ---
# 
# 1. **정형 데이터 읽기:** Local에 저장되어 있는 정형 데이터를 불러오고 확인
# 
# 2. **데이터 전처리:** 모델에 필요한 입력 형태로 데이터 처리
# 
# 3. **나이브 베이즈 분류기 정의 :** 나이브 베이즈 분류기  구현
# 
# 4. **하이퍼 파라미터 설정 및 컴파일 :** 올바른 하이퍼 파라미터 설정
# 
# 5. **모델 학습 수행:** <span style="color:red"> 나이브 베이즈 분류기  </span>을 통한 학습 수행, 평가 및 예측 수행
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
# |Run|Epochs|Scaler|frac|splitter|max_depth|min_split|min_leaf|criterion|splitter|max_depth|min_split|min_leaf|변수|lr|class 0|class 1|
# |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# |01|-|Standard|0.8||||||||||전부|-|0.965289|0.000000|
# |02|-|MinMax|0.8||||||||||전부|-|0.965289|0.000000|
# |03|-|MinMax|0.7||||||||||전부|-|에러|에러|
# |04|-|Robust|0.8||||||||||전부|-|0.40|0.13|
# 
# class 0 / class 1의 f1-score가 각각 0.26 / 0.13 이상이면 100점
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
# 
# - 결측값 처리

# ### 2.1 데이터 결측치 보간

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


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(20,20))

dist_cnt=0

def distplot(column_name):    
    plt.subplot(3, 3, dist_cnt)
    sns.distplot(df[column_name+'th col'], hist=True, rug=True, color='y')
    plt.title(f'{column_name}th Sensor Measurements', fontsize = 20)


for i in range(5, 14):
    dist_cnt += 1
    distplot(str(i))
    
plt.show()


# - 데이터의 상관 관계 분석

# ## 4. 데이터 전처리

# In[ ]:


labels = pd.read_csv("secom_labels.data", sep=" ", names = ['label', 'date'])


# In[ ]:


labels = labels.drop(['date'], axis=1)
labels = labels.replace(-1, 0)


# In[ ]:


labels


# In[ ]:


# 라벨의 비율 확인.
cnt_zero = 0
cnt_one = 0
for l in labels.values:
    if l == 1:
        cnt_one+=1
    else:
        cnt_zero+=1

print(f"label ratio -> {cnt_zero} : {cnt_one}")


# In[ ]:


# label 분포 시각화
labels_list = ['Pass', 'Fail']
values = [cnt_zero, cnt_one]
explode = (0, 0.1,)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(values, explode=explode, labels=labels_list, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


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
# st_scaler = MinMaxScaler()
st_scaler = RobustScaler()

normed_train_data = st_scaler.fit_transform(train_dataset)
normed_test_data = st_scaler.fit_transform(test_dataset)
normed_train_data = pd.DataFrame(normed_train_data)
normed_test_data = pd.DataFrame(normed_test_data)


# In[ ]:


# 0 인 데이터 제거할 때 유용한 코드
# df=df.loc[:,df.std()>0]


# In[ ]:


# def norm(x):
#     # 표준 스케일링 직접 구현하여 적용하면 NaN값 생성->결측 column 제거 손쉽게 가능

# normed_train_data = norm(train_dataset)
# normed_test_data = norm(test_dataset)


# In[ ]:


normed_train_data.head() # 스케일링 된 값 확인.


# In[ ]:


# 결측치 값 제거.
normed_train_data = normed_train_data.dropna(axis=1)
normed_test_data = normed_test_data.dropna(axis=1)


# In[ ]:


normed_train_data.shape


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


adasyn = ADASYN(random_state=0)
X_resampled, y_resampled = adasyn.fit_resample(normed_train_data, train_labels)


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


# In[ ]:


# label 분포 시각화
labels_list = ['Pass', 'Fail']
values = [cnt_zero, cnt_one]
explode = (0, 0.1,)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(values, explode=explode, labels=labels_list, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# ## 5. 머신러닝 모델

# ### 5.5 나이브 베이즈 (Naïve Bayes Classification) 분류기
# ----
# 
# - 데이터의 각 특징들이 독립적이라는 가정 아래에 베이즈 정리(Bayes Rule)를 이용하여 해당 범주에 속할 확률을 계산함.
# - 가우시안 나이브 베이즈 분류의 경우 연속형 변수(X) 에 대해 가우시안분포 확률밀도함수를 활용한 확률을 사용해 분류.

#  ![05_naive_bayes](02_02_file/05_naive_bayes.png)

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


naive_bayes_classifier = GaussianNB()


# #### 5.2.1 학습

# In[ ]:


# naive_bayes_classifier.fit(normed_train_data, train_labels.values.ravel())
naive_bayes_classifier.fit(normed_train_data, train_labels)


# #### 5.2.2 예측 및 평가
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


from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


prec_labels_naive = naive_bayes_classifier.predict(normed_test_data)


# In[ ]:


mean_acc = naive_bayes_classifier.score(normed_test_data, test_labels)


# In[ ]:


print(mean_acc)


# In[ ]:


print(classification_report(test_labels, prec_labels_naive))


# In[ ]:


from sklearn.metrics import plot_roc_curve


# In[ ]:


plot_roc_curve(naive_bayes_classifier, normed_test_data, test_labels)


# #### 5.1.3 샘플링 데이터로 학습

# In[ ]:


sampled_naive_bayes_classifier = GaussianNB()
sampled_naive_bayes_classifier.fit(X_resampled, y_resampled.values.ravel())


# In[ ]:


prec_labels_naive = sampled_naive_bayes_classifier.predict(normed_test_data)
print(classification_report(test_labels, prec_labels_naive ))


# In[ ]:


prec_labels_naive


# In[ ]:


mean_acc = sampled_naive_bayes_classifier.score(normed_test_data, test_labels)


# In[ ]:


mean_acc


# In[ ]:


plot_roc_curve(sampled_naive_bayes_classifier, normed_test_data, test_labels)


# ## 6. 제출

# ※ 과제 제출 관련하여 jupyter notebook 최상단의 `randomseed(=42)`를 절대 수정하시 마세요
# 
# ---
# 
# 나이브베이즈 분류 모델을 구현하여 표준 스케일링 된 테스트 데이터(**normed_test_data**)를 추론해보세요.
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
# `sklearn.metrics`의 `classification_report`을 통한 성능 측정으로 **class 0 / class 1의 f1-score**가 각각 **0.26 / 0.13** 이상이면 100점입니다.
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

answer_df = pd.DataFrame(prec_labels_naive)
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

