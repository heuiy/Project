#!/usr/bin/env python
# coding: utf-8

# # [Case1-2] 광석 처리 공단에서의 이산화규소 (silica) 불순도 예측_Rev.02

# ---

# 
# ## 프로젝트 목표
# ---
# - **회귀 나무 모델**을 구현.
# - 불필요한 입력값을 제거하는 데이터 정제 기법.
# - 데이터 스케일링 작업.
# - 학습, 평가, 검증 데이터 전처리 작업.
# - 회귀 나무 하이퍼파라미터 튜닝 기법.
# 

# ## 프로젝트 목차
# ---
# 
# 1. **정형 데이터 읽기:** Local에 저장되어 있는 Kaggle 데이터를 불러오고 데이터 프레임 확인
# 
# 2. **데이터 전처리:** 모델에 필요한 입력 형태로 데이터 처리
# 
# 3. **회귀나무 모델 정의 :** 회귀나무 모델 구현
# 
# 4. **하이퍼 파라미터 설정 및 컴파일 :** 올바른 하이퍼 파라미터 설정
# 
# 5. **모델 학습 수행:** <span style="color:red">회귀 나무</span>을 통한 학습 수행, 평가 및 예측 수행
# 
# 6. **제출:** 예측한 결과를 제출한 후 채점 결과를 확인합니다.

# ## 데이터 출처
# ---
# https://www.kaggle.com/edumagalhaes/quality-prediction-in-a-mining-process?select=MiningProcess_Flotation_Plant_Database.csv

# ## 프로젝트 개요
# ---
# 
# **데이터:** 실제 제조 공장, 특히 광산 공장의 데이터베이스로, 채굴 과정에서 가장 중요한 부분 중 하나 인 부유 플랜트 에서 나온 데이터.
# 
# **목표:** 주요 목표는 이 데이터를 사용하여 광석 정광에 얼마나 많은 불순물이 있는지 예측하는 것.
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

# ## 결과 요약
# 
# 각 파라미터 조건에 따른 모델 평가 결과는 아래와 같다.
# 
# |Run|Epochs|Scaler|나무_criterion|splitter|max_depth|min_split|min_leaf|앙상블_criterion|splitter|max_depth|min_split|min_leaf|변수|lr|MSE|MAE|
# |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# |01|X|Standard|mse|best|3|2|1|mse|best|3|2|1|전부|-|1.083430|0.830101|
# |02|X|Standard|mse|best|500|2|1|mse|best|3|2|1|전부|-|0.216234|0.136238|
# |03|X|Standard|mse|best|500|2|1|mse|best|10|2|1|전부|-|0.216234|0.136238|
# |04|X|Standard|mae|random|500|2|1|mse|best|10|2|1|전부|-|X|X|
# |05|X|Standard|mse|best|300|2|2|mse|best|300|2|2|전부|-|0.215041|0.142336|
# |06|X|Standard|mse|best|500|5|2|mse|best|500|5|2|전부|-|0.214499|0.142360|
# |07|X|Standard|mse|best|500|10|10|mse|best|500|10|10|전부|-|0.217745|0.177417|
# |**08**|X|Standard|mse|best|500|2|2|mse|random|500|2|2|전부|-|0.214461|0.140891|
# |09|X|Standard|mse|random|500|2|2|mse|random|3|2|1|전부|-|0.255476|0.181215|
# |10|X|Standard|mse|random|500|2|2|mse|random|3|2|1|2개제거|-|0.326285|0.222588|
# |11|X|MinMax|mse|random|500|2|2|mse|random|3|2|1|5개제거||0.631216|0.444633|
# 
# mse < 0.21, mae < 0.15가 100점
# 
# 목표 성능을 달성하기 위해서 아래 어떤 방법을 활용하여도 좋습니다.
# - 상관 관계 분석을 통한 feature selection
# - 학습 하이퍼 파라미터
#     - criterion
#     - splitter
#     - max_depth
#     - min_samples_split
#     - min_samples_leaf 조정
#     - 모델 앙상블 등

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


df.head(n=10)


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


# In[ ]:


df = df.dropna() # 결측치 제거.
df.shape # 결측치가 없어 기존과 동일.


# ### 3.2 데이터의 상관 관계 분석

# In[ ]:


SMALL_SIZE = 15
MEDIUM_SIZE = 20
WEIGHT = 'bold'
plt.rc('font', size=SMALL_SIZE, weight=WEIGHT) # controls default text sizes 
plt.rc('xtick', labelsize=MEDIUM_SIZE) # fontsize of the tick labels 
plt.rc('ytick', labelsize=MEDIUM_SIZE) # fontsize of the tick labels
# plt.figure(figsize=(30, 25))
# p = sns.heatmap(df.corr(), annot=True) # 데이터 컬럼 간, 상관관계 시각화. 하얀색이 더 높은 양의 상관관계.


# ### 3.3 산점도 시각화

# In[ ]:


# sns.set(rc = {'figure.figsize':(15,8)})
# sns.pairplot(df[['% Silica Concentrate', 'Starch Flow', 'Ore Pulp Density', 'Flotation Column 04 Level', 'Flotation Column 05 Level', 'Flotation Column 06 Level']], diag_kind='kde')


# ## 4. 데이터 전처리

# In[ ]:


# 필요없다고 판단되는 특성 몇 가지를 제거.
##### 실습 과제 수행시 상관관계와 산점도를 분석한 뒤 추가적으로 특성 제거

# df = df.drop(['date', '% Iron Concentrate'], axis=1)  # 실험실 측정 값으로 예측시엔 활용 불가하여 제거

# df = df.drop(['Starch Flow', 'Amina Flow'], axis=1)  # 실험실 측정 값으로 예측시엔 활용 불가하여 제거

df = df.drop(['Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density'], axis=1)  # 실험실 측정 값으로 예측시엔 활용 불가하여 제거


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


# ### 4.2 특성 스케일링

# In[ ]:


from sklearn.preprocessing import StandardScaler # 표준 스케일링  (평균 = 0 / 표준편차 = 1)
from sklearn.preprocessing import MinMaxScaler  # 최대/최소 스케일링 (이상치에 취약)
from sklearn.preprocessing import RobustScaler  # 중앙값 = 0 / IQR(1분위(25%) ~ 3분위(75%)) = 1 (이상치 영향 최소화, 넓게 분포)
from sklearn.preprocessing import MaxAbsScaler  # |x|  <= 1 , 이상치에 취약할 수 있다. 양수만 있는 데이터의 경우 MinMaxScaler 유사


# In[ ]:


# st_scaler = StandardScaler()
st_scaler = MinMaxScaler()

normed_train_data = st_scaler.fit_transform(train_dataset)
normed_test_data = st_scaler.fit_transform(test_dataset)
normed_train_data = pd.DataFrame(normed_train_data)
normed_test_data = pd.DataFrame(normed_test_data)


# In[ ]:


normed_train_data.head() # 스케일링 된 값 확인.


# ## 5. 머신러닝 모델 수행

# ### 5.1 회귀 나무
# 
# - 트리의 루트부터 시작해서 불순도가 가장 작아지는 특성 분할을 찾아나가는 모델.
# - 노드 내, MSE 또는 MAE 값을 불순도로 사용함.

# In[ ]:


# 데이터 분석 및 머신러닝을 위한 sklearn 라이브러리를 이용.
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


# 회귀 나무 파라미터 정의
criterion='mse'  # mse, friedman_mse, mae, poisson
splitter='random'  # best, random
max_depth=500
min_samples_split=2
min_samples_leaf=2

# 회귀 나무 정의
tree = DecisionTreeRegressor(max_depth=max_depth, 
                             criterion=criterion, 
                             splitter=splitter,
                             min_samples_split=min_samples_split, 
                             min_samples_leaf=min_samples_leaf)


# #### 5.1.1 학습

# In[ ]:


_ = tree.fit(normed_train_data, train_labels)


# #### 5.1.2 예측

# In[ ]:


predictions = tree.predict(normed_test_data)


# In[ ]:


print(f"prediction results : {predictions}")
print('='*100)
print(f"test labels: {test_labels}")


# #### 5.1.3 평가

# In[ ]:


# 회귀 분석 정확도 측정 - 잔차 계수 및 결정 계수
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


mse = mean_squared_error(test_labels, predictions)
mae = mean_absolute_error(test_labels, predictions) 

print("Test mse error: ", mse)
print("Test mae error: ", mae)


# #### 5.1.4 모델 앙상블

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.linear_model import SGDRegressor
# 회귀 나무 파라미터 정의
criterion='mse'  # mse, friedman_mse, mae, poisson
splitter='random'  # best, random
max_depth=3
min_samples_split=2
min_samples_leaf=1


# 두 개의 서로 다른 회귀 나무 정의  (1-1의linear regression model과 같은 다른 머신러닝 모델과도 같은 원리로 앙상블 가능)
reg_dt_1 = DecisionTreeRegressor(max_depth=max_depth, 
                             criterion=criterion, 
                             splitter=splitter,
                             min_samples_split=min_samples_split, 
                             min_samples_leaf=min_samples_leaf)

reg_dt_2 = DecisionTreeRegressor(max_depth=max_depth, 
                             criterion=criterion, 
                             splitter=splitter,
                             min_samples_split=min_samples_split, 
                             min_samples_leaf=min_samples_leaf)



estimators = [('dt_1', reg_dt_1),('dt_2', reg_dt_2)]


# reg_dt_1, reg_dt_2 두 모델 앙상블
reg_rating = VotingRegressor(estimators)
reg_rating.fit(normed_train_data, train_labels)

reg_pred = reg_rating.predict(normed_test_data)


# In[ ]:


from sklearn.metrics import mean_squared_error # MSE 로 오차 측정.
from sklearn.metrics import mean_absolute_error # MAE로 오차 측정.

# 예측모형 성능
mse = mean_squared_error(test_labels, reg_pred)
mae = mean_absolute_error(test_labels, reg_pred) 

print("Test mse error: ", mse)
print("Test mae error: ", mae)

print(f"prediction results: {reg_pred}")
print('='*100)
print(f"test labels: {test_labels}")


# ## 6. 제출

# ※ 과제 제출 관련하여 jupyter notebook 최상단의 `randomseed(=42)`를 절대 수정하시 마세요
# 
# ---
# 
# 
# 회귀 나무 모델을 구현하여 표준 스케일링 된 테스트 데이터(**normed_test_data**)를 추론해보세요
# 
# 목표 성능을 달성하기 위해서 `상관 관계 분석`을 통한 feature selection, 학습 하이퍼 파라미터(`criterion`, `splitter`, `max_depth`, `min_samples_split`, `min_samples_leaf`) 조정, `모델 앙상블` 등 어떤 방법을 활용하여도 좋습니다.
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
# `sklearn.metrics`의 `mean_squared_error`, `mean_absolute_error`을 통한 성능 측정으로 채점되며 **mse < 0.21, mae < 0.15**가 100점입니다 
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

