# 경로 문제 해결
    # 경로 내에 특수 문자 처리하도록 r 삽입
        # dwg_folder_path = r"D:\#.Secure Work Folder\BIG\Project\23~24Y\240129 CAD Reading Automation\DWG\in\240119 first"
    # DWG 파일을 DXF 파일로 변환할 때 shell=True 반영
        #    subprocess.run(command_string, shell=True)

import os
import subprocess

# DWG 파일이 있는 폴더 경로
dwg_folder_path = r"D:\#.Secure Work Folder\BIG\Project\23~24Y\240129 CAD Reading Automation\DWG\in\240119 first"

# 변환된 DXF 파일을 저장할 폴더 경로
dxf_folder_path = r"D:\#.Secure Work Folder\BIG\Project\23~24Y\240129 CAD Reading Automation\DXF\in\240119 first"

# Teigha File Converter 실행 파일 경로
teigha_converter_path = r"C:\Program Files (x86)\ODA\Teigha File Converter 4.3.2\TeighaFileConverter.exe"

# 폴더 내의 모든 DWG 파일을 순회
for filename in os.listdir(dwg_folder_path):
   if filename.lower().endswith(".dwg"):
       dwg_file_path = os.path.join(dwg_folder_path, filename)
       dxf_file_name = os.path.splitext(filename)[0] + ".dxf"
       dxf_file_path = os.path.join(dxf_folder_path, dxf_file_name)

       # Teigha File Converter의 명령줄 형식에 맞게 인자를 구성합니다.
       command = [
           f'"{teigha_converter_path}"',
           f'"{dwg_folder_path}"',
           f'"{dxf_folder_path}"',
           "ACAD2018",  # 또는 원하는 AutoCAD 버전을 명시
           "DXF",  # 출력 파일 형식
           "0",  # 하위 폴더를 재귀적으로 탐색하지 않음
           "1",  # 파일 검사 수행
           f'"{filename}"'  # 처리할 파일
       ]

       # 명령어를 문자열로 합칩니다.
       command_string = " ".join(command)

       # DWG 파일을 DXF 파일로 변환
       subprocess.run(command_string, shell=True)
