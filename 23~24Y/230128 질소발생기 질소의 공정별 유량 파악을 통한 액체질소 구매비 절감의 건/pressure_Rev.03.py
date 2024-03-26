# Timestamp 인식을 못해서 계속 오류 발생
# 23.11.29 이전은 parsing 에러 발생,
# 23.11.29 이후는 정상 parsing,

import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob

# 로컬 디렉토리 경로
input_folder = "D:/#.Secure Work Folder/BIG/Project/23~24Y/230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출/in/"

# 해당 폴더 내의 모든 엑셀 파일 가져오기
file_paths = glob.glob(input_folder + "*.xlsx")

dataframes = []

# 각 엑셀 파일을 DataFrame으로 불러오기
for file_path in file_paths:
  df = pd.read_excel(file_path)
  dataframes.append(df)

# 모든 데이터를 하나의 DataFrame으로 합치기
combined_df = pd.concat(dataframes, ignore_index=True)

# 타임스탬프 파싱 함수
def parse_timestamp(ts):
   if pd.isnull(ts):
       return np.nan
   # 데이터 타입 확인 및 문자열로 변환
   if not isinstance(ts, str):
       ts = str(ts)
   # 다양한 시간 형식을 시도
   formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %I:%M %p', '%Y-%m-%d %p %I:%M']  # 24시간제 형식 추가
   for fmt in formats:
       try:
           return datetime.strptime(ts, fmt)
       except ValueError as e:
           print(f"Timestamp 파싱 오류: {ts}, 오류: {e}")  # 오류 메시지 출력
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

  # 결측치 제거 및 전파
  sensor_df = sensor_df.dropna()
  sensor_df = sensor_df.ffill()  # 'ffill'을 사용하여 결측치를 앞의 값으로 채움

  # 처리된 센서 데이터 저장
  sensor_dataframes[timestamp_col] = sensor_df

# 각 센서 데이터 확인 (예시)
for sensor, df in sensor_dataframes.items():
  print(f"Sensor: {sensor}")
  print(df.head(), '\n')

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

# 결과를 CSV 파일로 저장
output_folder = "D:/#.Secure Work Folder/BIG/Project/23~24Y/230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출/output/"
output_file = "pressure.csv"
combined_df.to_csv(output_folder + output_file, index=False)