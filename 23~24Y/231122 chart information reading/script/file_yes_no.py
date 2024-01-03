import os

# 파일 경로 지정
pdf_path = 'D:/#.Secure Work Folder/BIG/Project/23~24Y/231122 chart information reading/pdf/in/ARE24008/20231117115139-0001.pdf'

# 파일 존재 여부 확인
if os.path.exists(pdf_path):
   print(f'파일이 존재합니다: {pdf_path}')
   # 파일 읽기 권한 확인
   if os.access(pdf_path, os.R_OK):
       print('파일에 접근할 수 있습니다.')
   else:
       print('파일에 접근할 수 없습니다. 권한을 확인하세요.')
else:
   print(f'파일이 존재하지 않습니다: {pdf_path}')