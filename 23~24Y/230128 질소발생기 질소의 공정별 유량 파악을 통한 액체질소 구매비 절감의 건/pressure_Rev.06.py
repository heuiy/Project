# Timestamp 정상 parsing 됨
# Timestamp 하나로 모으는 기능까지 추가
# 현재까지 Best

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

def parse_timestamp(ts):
   if pd.isnull(ts):
       return np.nan
   if not isinstance(ts, str):
       ts = str(ts)
   formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %I:%M %p', '%Y-%m-%d %p %I:%M']
   for fmt in formats:
       try:
           return datetime.strptime(ts, fmt)
       except ValueError:
           continue
   print(f"Timestamp 파싱 오류: {ts}")
   return np.nan

# 타임스탬프 칼럼을 식별
timestamp_cols = [col for col in combined_df.columns if 'PV_Timestamp' in col]

# 각 센서별로 데이터 처리
for timestamp_col in timestamp_cols:
   combined_df[timestamp_col] = combined_df[timestamp_col].apply(parse_timestamp)
   combined_df.sort_values(by=timestamp_col, inplace=True)

# Confidence 칼럼 제거
confidence_cols = [col for col in combined_df.columns if 'Confidence' in col]
combined_df.drop(confidence_cols, axis=1, inplace=True)

# 특정 값 필터링
combined_df = combined_df[combined_df['FI_S_105.PV_Value'] < 800]
combined_df = combined_df[combined_df['PI_S_205.PV_Value'] < 1.5]
combined_df = combined_df[combined_df['PI_S_220.PV_Value'] < 1.5]
combined_df = combined_df[combined_df['PI_S_241.PV_Value'] < 1.5]
combined_df = combined_df[combined_df['PI_S_301.PV_Value'] < 1.5]

# 타임스탬프 데이터프레임 생성
timestamps = []
for col in combined_df.columns:
   if 'Timestamp' in col:
       timestamps.extend(combined_df[col].astype(str).tolist())

timestamps = sorted(set(timestamps))

timestamp_df = pd.DataFrame(timestamps, columns=['Timestamp'])
timestamp_df['Timestamp'] = pd.to_datetime(timestamp_df['Timestamp'])
timestamp_df = timestamp_df.drop_duplicates().reset_index(drop=True)

# 각 센서 데이터 재구성 및 보간
for i in range(0, len(combined_df.columns) - 1, 2):
   sensor_df = combined_df[[combined_df.columns[i], combined_df.columns[i+1]]].copy()
   sensor_name = combined_df.columns[i].split('.')[0]  # 센서 이름 추출
   sensor_df.columns = ['Timestamp', sensor_name]  # 열 이름 변경
   sensor_df['Timestamp'] = pd.to_datetime(sensor_df['Timestamp'])
   sensor_df = sensor_df.dropna().drop_duplicates(subset='Timestamp').set_index('Timestamp')
   sensor_df = sensor_df.reindex(timestamp_df['Timestamp']).interpolate(method='linear').reset_index()
   timestamp_df = pd.merge(timestamp_df, sensor_df, on='Timestamp', how='left', suffixes=('', '_' + sensor_name))

# 신규 칼럼 추가
timestamp_df['SP'] = 70

# 결과 저장
output_folder = "D:/#.Secure Work Folder/BIG/Project/23~24Y/230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출/output/"
output_file = "pressure.csv"
timestamp_df.to_csv(output_folder + output_file, index=False)