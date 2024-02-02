# 경로 이슈로 잘 안됨

import os
import subprocess

# DWG 파일이 있는 폴더 경로
# dwg_folder_path = "D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\240129 CAD Reading Automation\\DWG\\in\\240119 first"
dwg_folder_path = "D:/#.Secure Work Folder/BIG/Project/23~24Y/240129 CAD Reading Automation/DWG/in/240119 first/"

# 변환된 DXF 파일을 저장할 폴더 경로
# dxf_folder_path = "D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\240129 CAD Reading Automation\\DXF\\in\\240119 first"
dxf_folder_path = "D:/#.Secure Work Folder/BIG/Project/23~24Y/240129 CAD Reading Automation/DXF/in/240119 first/"

# Teigha File Converter 실행 파일 경로
# teigha_converter_path = "C:\\Program Files (x86)\\ODA\\Teigha File Converter 4.3.2\\TeighaFileConverter.exe"
teigha_converter_path = "C:/Program Files (x86)/ODA/Teigha File Converter 4.3.2/TeighaFileConverter.exe"

# 폴더 내의 모든 DWG 파일을 순회
for filename in os.listdir(dwg_folder_path):
   if filename.lower().endswith(".dwg"):
       # 입력과 출력 경로를 큰따옴표로 감싸줍니다.
       quoted_input_folder = f'"{dwg_folder_path}"'
       quoted_output_folder = f'"{dxf_folder_path}"'

       # Teigha File Converter의 명령줄 형식에 맞게 인자를 구성합니다.
       command = [
           teigha_converter_path,
           quoted_input_folder,
           quoted_output_folder,
           "ACAD2020",  # 또는 원하는 AutoCAD 버전을 명시
           "DXF",  # 출력 파일 형식
           "0",  # 하위 폴더를 재귀적으로 탐색하지 않음
           "1",  # 파일 검사 수행
           "*.dwg"  # 처리할 파일 형식
       ]

       # DWG 파일을 DXF 파일로 변환
       subprocess.run(command)
