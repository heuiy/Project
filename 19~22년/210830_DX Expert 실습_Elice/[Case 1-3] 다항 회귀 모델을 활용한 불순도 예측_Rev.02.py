#!/usr/bin/env python
# coding: utf-8

# # [Case1-3] 광석 처리 공단에서의 이산화규소 (silica) 불순도 예측_Rev.02

# ---

# 
# ## 프로젝트 목표
# ---
# - **다항 회귀 분석 모델을 구현**.
# - 불필요한 입력값을 제거하는 데이터 정제 기법.
# - 데이터 스케일링 작업.
# - 학습, 평가, 검증 데이터 전처리 작업.
# - 다항 회귀 모델 하이퍼파라미터 튜닝 기법.
# - **다항 회귀 모델의 과적합 억제 기법**.
# 

# ## 프로젝트 목차
# ---
# 
# 1. **정형 데이터 읽기:** Local에 저장되어 있는 Kaggle 데이터를 불러오고 데이터 프레임 확인
# 
# 2. **데이터 전처리:** 모델에 필요한 입력 형태로 데이터 처리
# 
# 3. **다항회귀 모델 정의 :** 다항회귀 모델 구현
# 
# 4. **하이퍼 파라미터 설정 및 컴파일 :** 올바른 하이퍼 파라미터 설정
# 
# 5. **모델 학습 수행:** <span style="color:red">다항 회귀 모델</span>을 통한 학습 수행, 평가 및 예측 수행
# 
# 6. **제출:** 예측한 결과를 제출한 후 채점 결과를 확인합니다.
# 

# ## 데이터 출처
# ---
# https://www.kaggle.com/edumagalhaes/quality-prediction-in-a-mining-process?select=MiningProcess_Flotation_Plant_Database.csv

# ## 프로젝트 개요
# ---
# 
# **데이터:** 실제 제조 공장, 특히 광산 공장의 데이터베이스로, 채굴 과정에서 가장 중요한 부분 중 하나 인 부유 플랜트 에서 나온 데이터.
# 
# **목표:** 주요 목표는이 데이터를 사용하여 광석 정광에 얼마나 많은 불순물이 있는지 예측하는 것.
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
# |Run|regularization|early_stop|warm_start|max_iter|power_t|loss|min_leaf|max_depth|n_iter_no_change|lr_scheduling|lr|MSE|MAE|
# |-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# |01|L2|F|F|1|0.1|MSE|-|-|1|adaptive|0.0005|1.015747|0.783937|
# |02|elasticnet|T|F|1|0.01|MSE|-|-|1|adaptive|0.0001|0.943983|0.760109|
# |03|elasticnet|T|T|1|0.01|MSE|-|-|100|adaptive|0.0001|0.941813|0.752277|
# |04|elasticnet|T|T|100|0.01|MSE|-|-|100|constant|0.00001|0.927342|0.747354|
# |05|elasticnet|T|T|200|1|MSE|-|-|10|adaptive|0.00001|0.929414|0.750583|
# |06|elasticnet|T|T|200|0.01|MSE|-|-|1000|adaptive|0.00001|0.929799|0.750289|
# |07|L2|T|T|200|0.01|MSE|-|-|1000|constant|0.00001|0.926821|0.745555|
# |08|L1|T|T|500|0.01|MSE|-|-|1000|optimal|0.00001|1023505357.756091|19888.550095|
# |09|elasticnet|F|T|1|0.1|MSE|-|-|10|optimal|0.00001|폭발|폭발|
# |10|L2|F|F|1|0.1|huber|-|-|10|adaptive|0.00001|1.487591|0.890843|
# 
# mse < 1.0, mae < 0.73가 100점
# 
# * degree=3으로만 해도 죽습니다.
# * degree=2로 고정
# 
# 목표 성능을 달성하기 위해서 아래 어떤 방법을 활용하여도 좋습니다.
# - 상관 관계 분석을 통한 feature selection
# - 학습 하이퍼 파라미터
#     - loss
#     - penalty
#     - max_iter
#     - learning_rate_scheduling
#     - learning_rate
#     - n_iter_no_change
#     - warm_start 조정 등

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


plt.figure(figsize=(30, 25))
p = sns.heatmap(df.corr(), annot=True) # 데이터 컬럼 간, 상관관계 시각화. 하얀색이 더 높은 상관관계.


# ### 3.2 산점도 시각화

# In[ ]:


sns.pairplot(df[['% Silica Concentrate', 'Amina Flow', 'Ore Pulp pH', 'Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow', 'Flotation Column 03 Air Flow']], diag_kind='kde')


# ## 4. 데이터 전처리

# In[ ]:


# 필요없다고 판단되는 특성 몇 가지를 제거.
##### 실습 과제 수행시 상관관계와 산점도를 분석한 뒤 추가적으로 특성 제거

df = df.drop(['% Iron Concentrate'], axis=1)


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


# ## 5. 머신러닝 모델 수행

# ### 5.1 다항 회귀
# 
# 
# 다항 회귀를 위한 작업
# 
# - 회귀 모델을 다차원 다항식으로 두고 회귀 분석을 수행함.
# - sklearn 라이브러리의 PolynominalFeatures 를 통해 다항 회귀를 위한 n차항을 추가함.

# In[ ]:


# 데이터 분석 및 머신러닝을 위한 sklearn 라이브러리 이용.
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


# In[ ]:


# 2차항 추가
quadratic = PolynomialFeatures(degree=2)
quad_train_data = quadratic.fit_transform(normed_train_data)
quad_test_data = quadratic.fit_transform(normed_test_data)


# In[ ]:


# 다항회귀 모델 정의
# 다항회귀 모델도 다중 회귀식의 일종이므로 같은 모델을 사용.
reg_poly = LinearRegression()


# #### 5.1.1 학습

# In[ ]:


_ = reg_poly.fit(quad_train_data, train_labels)


# In[ ]:


predictions = reg_poly.predict(quad_test_data)


# #### 5.1.2 예측

# In[ ]:


print(f"prediction results : {predictions}")
print(f"labels : {test_labels}")


# #### 5.1.3 평가

# In[ ]:


from sklearn.metrics import mean_squared_error # MSE 
from sklearn.metrics import mean_absolute_error # MAE 


# In[ ]:


# 정답 값들과 평균 제곱오차(Mean Square Error) 및 평균 절대 오차(Mean Absolute Error)를 측정합니다.
error_mse = mean_squared_error(test_labels, predictions) 
error_mae = mean_absolute_error(test_labels, predictions)


# In[ ]:


print(f"Test mse error : {error_mse}")
print(f"Test mae error : {error_mae}")


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


# #### 5.1.4 학습 - sgd 사용
# ----
# - stochastic gradient descent 방법을 사용하여 다중 회귀 모델을 학습.
# - 모델의 loss function을 최소로 만드는 파라미터를 최적화함.
# - 학습률(learning rate) 및 학습 횟수(max_iter)와 같은 하이퍼 파라미터를 직접 조정하며 모델을 튜닝하고 평가.

# In[ ]:


from sklearn.linear_model import SGDRegressor
from sklearn.metrics  import mean_squared_error # MSE 
from sklearn.metrics import mean_absolute_error # MAE 


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

print(quad_train_data)
scaler.fit(quad_train_data)
scaled = scaler.transform(quad_train_data)


# In[ ]:


loss='huber'  # squared_loss, huber, epsilon_insensitive, squared_epsilon_insensitive
regularization='l2' # normalization: None,'l1', 'l2', 'elasticnet'
max_iter=1
learning_rate=0.00001 # LR value
learning_rate_scheduling='adaptive'  # constant, optimal, invscaling, adaptive
n_iter_no_change=1
warm_start=False
early_stopping=False
verbose=True


# In[ ]:


reg_sgd = SGDRegressor(loss=loss, penalty=regularization,
                       max_iter=max_iter,
                       learning_rate=learning_rate_scheduling, 
                       eta0=learning_rate,
                       n_iter_no_change=n_iter_no_change, 
                       warm_start=warm_start,
                       verbose=verbose, 
                       early_stopping=early_stopping,
                       power_t=0.1)
                      
                      
# 모델 학습
_ = reg_sgd.fit(quad_train_data, train_labels)


# In[ ]:


# predicitons_sgd = reg_sgd.predict(normed_test_data)
predicitons_sgd = reg_sgd.predict(quad_test_data)
print(predicitons_sgd)


# In[ ]:


mse = mean_squared_error(test_labels, predicitons_sgd)
mae = mean_absolute_error(test_labels, predicitons_sgd) 


# In[ ]:


print("Test mse error: ", mse)
print("Test mae error: ", mae)


# In[ ]:


# target과 prediction 사이 비교 시각화

palette = ['#FB6542', '#FFBB00', '#3F681C', '#375E97']

ci = 1.96*np.std(predicitons_sgd)/np.mean(predicitons_sgd)

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(18,6))
ax.plot(test_date, test_labels, color=palette[1], label='Actual Value')
ax.plot(test_date, predicitons_sgd, color=palette[2], label='Forecast')

ax.fill_between(test_date, (predicitons_sgd-ci), (predicitons_sgd+ci), color=palette[3],
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
# 다항 회귀의 모델을 구현하여 표준 스케일링 된 테스트 데이터(**normed_test_data**)를 추론해보세요
# 
# 목표 성능을 달성하기 위해서 `상관 관계 분석`을 통한 feature selection, 학습 하이퍼 파라미터(`loss`, `penalty`, `max_iter`, `learning_rate_scheduling`, `learning_rate`, `n_iter_no_change`, `warm_start`) 조정 등 어떤 방법을 활용하여도 좋습니다.
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
# `sklearn.metrics`의 `mean_squared_error`, `mean_absolute_error`을 통한 성능 측정으로 채점되며 **mse < 1.0, mae < 0.73**가 100점입니다 
# 
# (부분 점수 있음).

# ### 채점

# 결과 csv 파일을 저장 후, 아래 코드를 실행하면 채점을 받을 수 있습니다.
# 
# **아래 코드를 수정하면 채점이 불가능 합니다.**

# In[ ]:


# 제출할 dataframe을 아래 코드에 대입하여 submission.csv 파일로 저장합니다.

answer_df = pd.DataFrame(predicitons_sgd)
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

