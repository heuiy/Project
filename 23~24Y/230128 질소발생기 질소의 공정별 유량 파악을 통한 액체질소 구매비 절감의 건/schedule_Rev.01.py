# 엑셀 데이터 입력하면 df 으로 변환해줌

import pandas as pd
import numpy as np

# 엑셀 파일 로드
file_path = 'D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출\\schedule\\스케쥴.xlsx'
df = pd.read_excel(file_path)

# 전체 시간대에 대한 빈 데이터프레임 생성
start_date = df.min().min()  # 시작 날짜
end_date = df.max().max()    # 종료 날짜
all_dates = pd.date_range(start=start_date, end=end_date, freq='H')
final_df = pd.DataFrame(index=all_dates, columns=['DP60_FD', 'DP67_FD', 'DP72_FD-3', 'DP72_FD-4'])
final_df.fillna(0, inplace=True)

# 시간대 표시 함수
def mark_times(final_df, start_dates, end_dates, column_name, start_hour, end_hour):
   for start, end in zip(start_dates, end_dates):
       start_time = pd.Timestamp(start).replace(hour=start_hour)
       end_time = pd.Timestamp(end).replace(hour=end_hour)
       final_df.loc[start_time:end_time, column_name] = 1
   return final_df

# 각 FD에 대해 시간대 표시
fd_list = [('DP60_FD', 9, 21), ('DP67_FD', 9, 21), ('DP72_FD-3', 21, 21), ('DP72_FD-4', 21, 21)]

for fd, start_hour, end_hour in fd_list:
   start_col = f'{fd} start {start_hour:02d}'  # '09'와 같이 두 자리 형식으로 맞춤
   end_col = f'{fd} end {end_hour:02d}'        # '21'과 같이 두 자리 형식으로 맞춤
   final_df = mark_times(final_df, df[start_col], df[end_col], fd, start_hour, end_hour)

# 결과 출력
print(final_df)

# 아래 코드는 없어도 되긴 함

# 저장할 파일의 경로를 지정합니다.
output_file_path = 'D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\230128 질소 사용량 패턴 분석을 통한 질소 절감 방안 도출\\output\\final_df.csv'

# CSV 파일로 저장합니다.
final_df.to_csv(output_file_path)

print(f"파일이 성공적으로 저장되었습니다: {output_file_path}")
