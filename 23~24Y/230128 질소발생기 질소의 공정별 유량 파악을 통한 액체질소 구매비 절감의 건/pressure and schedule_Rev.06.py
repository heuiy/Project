# 여기서부터는 schedule.py 내용을 반영함
# Time Stamp 전부 통합
# Time Stamp 보간 (평균) 적용함
# 생산일정 반영하려니 15시, 16시 등 정시에만 0 또는 1이 나옴

import pandas as pd

# pressure.csv 파일 불러오기
pressure_file = 'D:/#.Secure Work Folder/BIG/Project/23~24Y/230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출/output/pressure.csv'
pressure_df = pd.read_csv(pressure_file)

# 스케쥴.xlsx 파일 불러오기
schedule_file = 'D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출\\schedule\\스케쥴.xlsx'
schedule_df = pd.read_excel(schedule_file)

# 스케쥴 데이터 프레임 처리
start_date = schedule_df.min().min()
end_date = schedule_df.max().max()
all_dates = pd.date_range(start=start_date, end=end_date, freq='H')
final_df = pd.DataFrame(index=all_dates)
final_df = final_df.reset_index().rename(columns={'index': 'Timestamp'})
final_df[['DP60_FD_run', 'DP67_FD_run', 'DP72_FD-3_run', 'DP72_FD-4_run']] = 0

# 시간대 표시 함수
def mark_times(final_df, start_dates, end_dates, column_name, start_hour, end_hour):
   for start, end in zip(start_dates, end_dates):
       start_time = pd.Timestamp(start).replace(hour=start_hour)
       end_time = pd.Timestamp(end).replace(hour=end_hour)
       mask = (final_df['Timestamp'] >= start_time) & (final_df['Timestamp'] <= end_time)
       final_df.loc[mask, column_name] = 1
   return final_df

# 스케쥴 열을 반복 처리
fd_list = [('DP60_FD_run', 9, 21), ('DP67_FD_run', 9, 21), ('DP72_FD-3_run', 21, 21), ('DP72_FD-4_run', 21, 21)]
for fd, start_hour, end_hour in fd_list:
   start_col = f'{fd.split("_run")[0]} start {start_hour:02d}'
   end_col = f'{fd.split("_run")[0]} end {end_hour:02d}'
   if start_col in schedule_df and end_col in schedule_df:
       final_df = mark_times(final_df, schedule_df[start_col], schedule_df[end_col], fd, start_hour, end_hour)

# 최종 데이터프레임 통합
pressure_df['Timestamp'] = pd.to_datetime(pressure_df['Timestamp'])
combined_df = pd.merge(pressure_df, final_df, on='Timestamp', how='left')

# 신규 칼럼 추가
combined_df['SP'] = 70  # 모든 값에 70 할당

# 결과 파일 저장
output_file = 'D:/#.Secure Work Folder/BIG/Project/23~24Y/230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출/output/combined.csv'
combined_df.to_csv(output_file, index=False)

# 결과 확인
print(combined_df.head())
