# 원래 코랩 코드

from datetime import datetime
import pandas as pd
import numpy as np
import requests
from googleapiclient.discovery import build
from google.colab import auth
import io

"""# 데이터 로딩"""

# 구글 시트 활용
  # 엑셀 파일을 통째로 Ctrl + C / V
# 230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출 > data
  # 해당 폴더에 다른 구글 시트 넣으면 안됨
  # 각 월별 n 개의 엑셀 데이터를 일괄적으로 코랩으로 불러옴
  # n 개의 데이터를 하나의 df 으로 합하기

# 구글 인증
auth.authenticate_user()

# 서비스 생성
drive_service = build('drive', 'v3')
sheets_service = build('sheets', 'v4')

# 230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출 > data
# https://drive.google.com/drive/folders/1wu_cNdeN6d2R9r1FVCwnbBftzjij-UvF?usp=drive_link

# 폴더 ID
folder_id = '1wu_cNdeN6d2R9r1FVCwnbBftzjij-UvF'

# 폴더 내의 모든 구글 시트 파일 가져오기
results = drive_service.files().list(
 q=f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet'",
 fields="files(id, name)").execute()
items = results.get('files', [])

# 파일을 로컬에 저장하고 DataFrame으로 불러오기 위한 준비
dataframes = []

for item in items:
 spreadsheet_id = item['id']
 sheet_metadata = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
 sheet_names = [s['properties']['title'] for s in sheet_metadata.get('sheets', '')]

 for sheet_name in sheet_names:
     # 시트 데이터를 CSV 형식으로 가져오기
     url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
     res = requests.get(url)

     # CSV 데이터를 DataFrame으로 변환
     df = pd.read_csv(io.StringIO(res.text))
     dataframes.append(df)

# 모든 데이터를 하나의 DataFrame으로 합치기
combined_df = pd.concat(dataframes, ignore_index=True)

combined_df

combined_df.columns

# 타임스탬프 파싱 함수
def parse_timestamp(ts):
 if pd.isnull(ts):
     return np.nan
 if isinstance(ts, str):
     try:
         return datetime.strptime(ts, '%Y-%m-%d %p %I:%M')
     except ValueError:
         return np.nan
 return np.nan

# 타임스탬프 칼럼을 식별
timestamp_cols = [col for col in combined_df.columns if 'PV_Timestamp' in col]

# 개별 센서 데이터를 처리하기 위한 딕셔너리 초기화
sensor_dataframes = {}

# 각 센서별로 데이터 처리
for timestamp_col in timestamp_cols:
 # 타임스탬프 파싱 및 정렬
 combined_df[timestamp_col] = combined_df[timestamp_col].apply(parse_timestamp)
 combined_df.sort_values(by=timestamp_col, inplace=True)

 # 센서 값 칼럼 이름 추출 (예: 'FI_S_105.PV_Value')
 value_col = timestamp_col.replace('Timestamp', 'Value')

 # 해당 센서의 타임스탬프와 값 칼럼만 선택
 sensor_df = combined_df[[timestamp_col, value_col]]
 sensor_df.dropna(inplace=True)

 # 결측치 처리
 sensor_df.fillna(method='ffill', inplace=True)

 # 처리된 센서 데이터 저장
 sensor_dataframes[timestamp_col] = sensor_df

# 각 센서 데이터 확인 (예시)
for sensor, df in sensor_dataframes.items():
 print(f"Sensor: {sensor}")
 print(df.head(), '\n')

"""## 불필요한 칼럼 제거"""

# Confidence 칼럼 식별
confidence_cols = [col for col in combined_df.columns if 'Confidence' in col]

# Confidence 칼럼 제거
combined_df.drop(confidence_cols, axis=1, inplace=True)

# FI_S_105.PV_Value 칼럼에서 800 이상인 값 제거
combined_df = combined_df[combined_df['FI_S_105.PV_Value'] < 800]

# PI_S_205.PV_Value, PI_S_220.PV_Value, PI_S_241.PV_Value, PI_S_301.PV_Value 칼럼에서 1.5 이상인 값 제거
combined_df = combined_df[combined_df['PI_S_205.PV_Value'] < 1.5]
combined_df = combined_df[combined_df['PI_S_220.PV_Value'] < 1.5]
combined_df = combined_df[combined_df['PI_S_241.PV_Value'] < 1.5]
combined_df = combined_df[combined_df['PI_S_301.PV_Value'] < 1.5]

# 신규 칼럼 추가
combined_df['DP60'] = (combined_df['PI_S_205.PV_Value'] >= 0.5).astype(int)
combined_df['DP67'] = (combined_df['PI_S_220.PV_Value'] >= 0.5).astype(int)
combined_df['DP72_FD3'] = (combined_df['PI_S_241.PV_Value'] >= 0.5).astype(int)
combined_df['DP72_FD4'] = (combined_df['PI_S_301.PV_Value'] >= 0.5).astype(int)
combined_df['SP'] = 70  # 모든 값에 70 할당

# 데이터 확인
print(combined_df.head())

"""## df 내보내기"""

from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/MyDrive/회사_demitri/BIG/Project/23~24Y/230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출/stream_df/nov.csv"

combined_df.to_csv(path, index=False)

import matplotlib.pyplot as plt

# 그래프 크기 설정
plt.figure(figsize=(20, 15))

# combined_df의 모든 수치형 칼럼에 대해 반복
for column in combined_df.select_dtypes(include=['number']).columns:
 plt.plot(combined_df[column], label=column)

# 그래프 제목 및 축 레이블 설정
plt.title('All Sensor Data Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Value')

# 범례 표시 (그래프가 많기 때문에 범례 위치 조정을 고려할 수 있음)
plt.legend(loc='upper left')

# 그래프 표시
plt.show()

"""## 특정 기간 그래프 추출"""

# 관심 있는 기간 설정
start_date = pd.to_datetime('2023-10-09')
end_date = pd.to_datetime('2023-10-22')

# 센서 데이터프레임 필터링
filtered_dataframes = {}
for timestamp_col, df in sensor_dataframes.items():
 filtered_df = df[(df[timestamp_col] >= start_date) & (df[timestamp_col] <= end_date)]
 filtered_dataframes[timestamp_col] = filtered_df

plt.figure(figsize=(15, 10))

for timestamp_col, sensor_df in filtered_dataframes.items():
 value_col = timestamp_col.replace('Timestamp', 'Value')
 plt.plot(sensor_df[timestamp_col], sensor_df[value_col], label=value_col)

plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Sensor Values Over Time (Filtered)')
plt.legend()
plt.show()

"""# 각 센서들의 데이터 분포 확인"""

# PV_Value 칼럼 식별
value_cols = [col for col in combined_df.columns if 'PV_Value' in col]

# PV_Value 칼럼에 대한 요약 통계 출력
description = combined_df[value_cols].describe()
print(description)

"""## 상자그래프"""

import matplotlib.pyplot as plt
import seaborn as sns

# FI Value 칼럼 식별
fi_value_cols = [col for col in combined_df.columns if 'FI_S' in col and 'PV_Value' in col]

# PI Value 칼럼 식별
pi_value_cols = [col for col in combined_df.columns if 'PI_S' in col and 'PV_Value' in col]

# FI Value 박스 플롯 설정
plt.figure(figsize=(12, 6))
combined_df[fi_value_cols].boxplot()

# 그래프 제목 및 축 레이블 설정
plt.title('Boxplot of FI Value Columns')
plt.ylabel('Value')
plt.xticks(rotation=45)  # x축 레이블 회전

# 그래프 표시
plt.show()

# PI Value 박스 플롯 설정
plt.figure(figsize=(12, 6))
combined_df[pi_value_cols].boxplot()

# 그래프 제목 및 축 레이블 설정
plt.title('Boxplot of PI Value Columns')
plt.ylabel('Value')
plt.xticks(rotation=45)  # x축 레이블 회전

# 그래프 표시
plt.show()

"""# 회귀모델"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

"""## 독립변수, 종속변수"""

combined_df.columns

# 독립 변수와 종속 변수 설정
X = combined_df[['FI_S_105.PV_Value',
               'PI_S_205.PV_Value', 'PI_S_220.PV_Value', 'PI_S_241.PV_Value', 'PI_S_301.PV_Value',
              'DP60', 'DP67', 'DP72_FD3', 'DP72_FD4', 'SP'
               ]]  # 독립 변수
y = combined_df[['FI_S_106.PV_Value', 'FI_S_107.PV_Value',
              'FI_S_108.PV_Value', 'FI_S_109.PV_Value',
              'FI_S_110.PV_Value']]  # 종속 변수

# 데이터를 학습 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1303)

"""## 회귀 모델"""

# 회귀 모델 생성 및 학습
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 학습된 모델로 테스트 데이터에 대한 예측 수행
y_pred_linear = linear_model.predict(X_test)

"""### MSE"""

# 성능 평가
linear_mse = mean_squared_error(y_test, y_pred_linear)
print(f"Linear Regression MSE: {linear_mse}")

"""## 랜덤 포레스트"""

# 랜덤 포레스트 모델 생성 및 학습
rf_model = RandomForestRegressor(n_estimators=100, random_state=1303)
rf_model.fit(X_train, y_train)

# 학습된 모델로 테스트 데이터에 대한 예측 수행
y_pred_rf = rf_model.predict(X_test)

"""### MSE"""

# 성능 평가
rf_mse = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest MSE: {rf_mse}")

"""# 예측"""

combined_df.columns

# # 독립 변수와 종속 변수 설정
# X = combined_df[['FI_S_105.PV_Value',
#                  'PI_S_205.PV_Value', 'PI_S_220.PV_Value', 'PI_S_241.PV_Value', 'PI_S_301.PV_Value',
#                 'DP60', 'DP67', 'DP72_FD3', 'DP72_FD4', 'SP'
#                  ]]  # 독립 변수
# y = combined_df[['FI_S_106.PV_Value', 'FI_S_107.PV_Value',
#                 'FI_S_108.PV_Value', 'FI_S_109.PV_Value',
#                 'FI_S_110.PV_Value']]  # 종속 변수

X_median = np.median(X, axis=0).reshape(1, -1)

# 회귀 모델을 사용하여 예측
y_pred_linear = linear_model.predict(X_median)
print(f"Predicted y value (Linear Regression) using medians of X: {y_pred_linear}")

# 랜덤 포레스트 모델을 사용하여 예측
y_pred_rf = rf_model.predict(X_median)
print(f"Predicted y value (Random Forest) using medians of X: {y_pred_rf}")