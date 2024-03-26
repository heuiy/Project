# Time Stamp 보간 (평균) 적용함
# schedule_df 합하는 것이 쉽지 않음
# 계속 에러남
# 절반의 성공

import pandas as pd
import numpy as np

# 파일 경로
pressure_path = "D:/#.Secure Work Folder/BIG/Project/23~24Y/230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출/output/pressure.csv"
schedule_path = "D:/#.Secure Work Folder/BIG/Project/23~24Y/230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출/output/schedule.csv"

# 파일 읽기
pressure_df = pd.read_csv(pressure_path)
schedule_df = pd.read_csv(schedule_path)

# schedule_df의 'Timestamp' 열을 datetime으로 변환
schedule_df['Timestamp'] = pd.to_datetime(schedule_df['Timestamp'])

# pressure_df의 모든 타임스탬프 열을 하나의 리스트로 통합
timestamps = []
for col in pressure_df.columns:
   if 'Timestamp' in col:
       timestamps.extend(pressure_df[col].astype(str).tolist())

timestamps = sorted(set(timestamps))

# 타임스탬프 데이터프레임 생성 및 중복 제거
timestamp_df = pd.DataFrame(timestamps, columns=['Timestamp'])
timestamp_df['Timestamp'] = pd.to_datetime(timestamp_df['Timestamp'])
timestamp_df = timestamp_df.drop_duplicates().reset_index(drop=True)

# 각 센서 데이터 재구성 및 보간
for i in range(0, len(pressure_df.columns) - 1, 2):
   sensor_df = pressure_df[[pressure_df.columns[i], pressure_df.columns[i+1]]].copy()
   sensor_name = pressure_df.columns[i].split('.')[0]  # 센서 이름 추출
   sensor_df.columns = ['Timestamp', sensor_name]  # 열 이름 변경
   sensor_df['Timestamp'] = pd.to_datetime(sensor_df['Timestamp'])
   sensor_df = sensor_df.dropna().drop_duplicates(subset='Timestamp').set_index('Timestamp')
   sensor_df = sensor_df.reindex(timestamp_df['Timestamp']).interpolate(method='linear').reset_index()
   timestamp_df = pd.merge(timestamp_df, sensor_df, on='Timestamp', how='left', suffixes=('', '_'+sensor_name))

# schedule_df와 합치기
combined_df = pd.merge(timestamp_df, schedule_df, on='Timestamp', how='left')

# 결과 확인
print(combined_df.head())

# 필요한 경우 combined_df 저장
combined_df.to_csv("D:/#.Secure Work Folder/BIG/Project/23~24Y/230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출/output/combined_data.csv")

