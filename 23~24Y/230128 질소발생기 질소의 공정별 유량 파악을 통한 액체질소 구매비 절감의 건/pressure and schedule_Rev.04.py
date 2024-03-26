# 여기서부터는 schedule.py 내용을 반영함
# Time Stamp 전부 통합
# Time Stamp 보간 (평균) 적용함
# 생산일정 반영하고 있음
# 잘 안됨. 스케쥴 파일 반영했더니 공란이 됨

import pandas as pd
import numpy as np

# 파일 경로
pressure_path = "D:/#.Secure Work Folder/BIG/Project/23~24Y/230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출/output/pressure.csv"

# 파일 읽기
pressure_df = pd.read_csv(pressure_path)

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

# 신규 칼럼 추가
combined_df = timestamp_df.copy()
combined_df['SP'] = 70  # 모든 값에 70 할당

# 생산일정 처리
file_path = 'D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출\\schedule\\스케쥴.xlsx'
df = pd.read_excel(file_path)

start_date = df.min().min()
end_date = df.max().max()
all_dates = pd.date_range(start=start_date, end=end_date, freq='H')
final_df = pd.DataFrame(index=all_dates)
final_df = final_df.reset_index().rename(columns={'index': 'Timestamp'})
final_df[['DP60_FD_run', 'DP67_FD_run', 'DP72_FD-3_run', 'DP72_FD-4_run']] = 0

# 시간대 표시 함수
def mark_times(final_df, start_dates, end_dates, column_name, start_hour, end_hour):
   for start, end in zip(start_dates, end_dates):
       start_time = pd.Timestamp(start).replace(hour=start_hour)
       end_time = pd.Timestamp(end).replace(hour=end_hour)
       final_df.loc[(final_df['Timestamp'] >= start_time) & (final_df['Timestamp'] <= end_time), column_name] = 1
   return final_df

fd_list = [('DP60_FD_run', 9, 21), ('DP67_FD_run', 9, 21), ('DP72_FD-3_run', 21, 21), ('DP72_FD-4_run', 21, 21)]

for fd, start_hour, end_hour in fd_list:
   start_col = f'{fd.split("_run")[0]} start {start_hour:02d}'
   end_col = f'{fd.split("_run")[0]} end {end_hour:02d}'
   final_df = mark_times(final_df, df[start_col], df[end_col], fd, start_hour, end_hour)

# 최종 데이터프레임 통합
combined_df = pd.merge(combined_df, final_df[['Timestamp', 'DP60_FD_run', 'DP67_FD_run', 'DP72_FD-3_run', 'DP72_FD-4_run']], on='Timestamp', how='left')

# 결과 확인
print(combined_df.head())

# 필요한 경우 combined_df 저장
combined_df.to_csv("D:/#.Secure Work Folder/BIG/Project/23~24Y/230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출/output/combined_data.csv")

