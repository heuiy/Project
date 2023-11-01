# 전혀 검출을 못하고 있음

# 중복되는 원 제거
# 문제 : 많은 원이 막 겹쳐짐
# 검출된 원들 사이의 거리를 계산하여 가까운 원을 제거

# 보라색 원 찾기
# HSV 색공간에서 보라색 범위에 속하는 픽셀만을 대상으로 원을 찾음
# 원 내부에서 여러 픽셀의 색상을 샘플링하여 평균을 구함
    # 문제 : 경계선만 보라색이어도 검출됨

# 중복 원 제거:
# unique_circles 배열을 사용하여 중복 원 제거
# 여전히 작은 원과 큰 원이 중복으로 검출
# 거리 뿐만 아니라 반지름도 고려하여 중복 원을 제거함

# 애매한 색상 판별:
# 보라색과 투명색을 더 정확하게 구분하려면,
# 각 원의 중심에서 색상을 샘플링하는 대신
# 원 내부의 여러 픽셀을 샘플링하여 평균 색상을 구함.
# 이 평균 색상을 사용하여 원의 색상을 판별

import cv2
import numpy as np
import os
import glob

# 입력 폴더 경로를 하드코딩합니다.
input_base_folder = "D:/#.Secure Work Folder/BIG/Project/00Temp/230829_CPE/in/"
output_base_folder = "D:/#.Secure Work Folder/BIG/Project/00Temp/230829_CPE/out/"

# 입력 폴더 내의 하위 폴더 목록을 출력하고 사용자에게 선택하게 합니다.
subfolders = [f.name for f in os.scandir(input_base_folder) if f.is_dir()]
for i, folder_name in enumerate(subfolders):
    print(f"{i+1}. {folder_name}")

selected_index = int(input("Please select a folder by entering its number: ")) - 1
selected_folder = subfolders[selected_index]

# 선택된 폴더로 작업 폴더를 설정합니다.
current_input_folder = os.path.join(input_base_folder, selected_folder)

# 결과를 저장할 출력 폴더를 생성합니다.
current_output_folder = os.path.join(output_base_folder, selected_folder)
os.makedirs(current_output_folder, exist_ok=True)

# 선택된 폴더 내의 모든 jpg 파일에 대해
for filepath in glob.glob(os.path.join(current_input_folder, "*.jpg")):
    img = cv2.imread(filepath)

    # 이미지 전처리
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 보라색과 파란색 범위 설정 및 마스크 생성 (HSV 채도, 명도 값 조정)
    lower_color = np.array([90, 50, 50])  # H value from 100 to include 파랑
    upper_color = np.array([170, 255, 255])  # H value to 160 to include 보라
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
    #                            param1=50, param2=30, minRadius=10, maxRadius=50)
    # minDist: 60으로 설정하여 원 간의 최소 거리를 늘렸습니다. 이 값이 작으면 중복해서 막 detect 됨
    # minRadius: 이 값보다 작은 반지름을 가진 원은 무시됩니다.
    # maxRadius: 이 값보다 큰 반지름을 가진 원은 무시됩니다.

    # 원 검출
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=60,
                               param1=50, param2=30, minRadius=130, maxRadius=220)

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        row_count = [0]*8  # 8개의 행에 대한 카운터

        # 원 중복 제거
        unique_circles = []
        for circle in circles:
            is_unique = True
            for unique_circle in unique_circles:
                distance = np.linalg.norm(circle[:2] - unique_circle[:2])
                radius_diff = abs(circle[2] - unique_circle[2])

                if distance < 30 and radius_diff < 20:
                    is_unique = False
                    break

            if is_unique:
                unique_circles.append(circle)

        for i in unique_circles:
            # 원 내부에서 여러 픽셀의 색상을 샘플링하여 평균을 구합니다.
            circle_color = np.mean([mask[y, x] for x in range(i[0] - 5, i[0] + 6) for y in range(i[1] - 5, i[1] + 6)])

            # 원이 속한 행을 찾습니다.
            row = i[1] // (img.shape[0] // 8)

            # 애매한 보라색도 보라색으로 판별
            if circle_color > 50:
                row_count[row] += 1
                cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 255), 8)  # 보라색 원을 굵게 그립니다.
            else:
                cv2.circle(img, (i[0], i[1]), i[2], (255, 255, 255), 8)  # 투명한 원을 굵게 그립니다.

        # 카운트 결과를 이미지 우측에 워터마크로 표시합니다.
        for idx, count in enumerate(row_count):
            watermark_text = f"{count}/12"
            cv2.putText(img, watermark_text, (img.shape[1] - 300, (img.shape[0] // 8) * idx + 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 8)

    # 결과 이미지 저장
    output_filepath = os.path.join(current_output_folder, os.path.basename(filepath))
    cv2.imwrite(output_filepath, img)