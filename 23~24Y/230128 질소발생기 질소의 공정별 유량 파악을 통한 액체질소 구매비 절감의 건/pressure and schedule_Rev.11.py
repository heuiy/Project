# pressure.csv 와 schedule.csv 두 개의 파일을 합치기
# 새롭게 시작!!!
# 현재까지 Best

import pandas as pd

# 파일 경로
pressure_path = "D:/#.Secure Work Folder/BIG/Project/23~24Y/230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출/output/pressure.csv"
schedule_path = "D:/#.Secure Work Folder/BIG/Project/23~24Y/230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출/output/schedule.csv"

# 파일 읽기
pressure_df = pd.read_csv(pressure_path)
schedule_df = pd.read_csv(schedule_path)

# Timestamp 컬럼을 datetime 타입으로 변환
pressure_df['Timestamp'] = pd.to_datetime(pressure_df['Timestamp'])
schedule_df['Timestamp'] = pd.to_datetime(schedule_df['Timestamp'])

# schedule_df의 각 행을 반복하며 pressure_df에 값을 추가
for i in range(len(schedule_df)):
   start_time = schedule_df.loc[i, 'Timestamp']
   # 마지막 행이 아니라면 다음 행의 Timestamp를 종료 시간으로 설정
   end_time = schedule_df.loc[i + 1, 'Timestamp'] if i < len(schedule_df) - 1 else pressure_df['Timestamp'].max()

   # 해당 시간대에 pressure_df의 행을 찾고 schedule_df의 값을 복사
   mask = (pressure_df['Timestamp'] >= start_time) & (pressure_df['Timestamp'] < end_time)
   for col in schedule_df.columns.drop('Timestamp'):
       pressure_df.loc[mask, col] = schedule_df.loc[i, col]

# 나머지 열에 대해서만 fillna 적용
cols_to_fill = pressure_df.columns.drop('Timestamp')
pressure_df[cols_to_fill] = pressure_df[cols_to_fill].fillna(0)

# 결과 저장
pressure_df.to_csv("D:/#.Secure Work Folder/BIG/Project/23~24Y/230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출/output/combined.csv", index=False)
