# 검출 잘 안됨

# 여기에 원에 하나씩 번호를 넣고 싶음

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
    # blurred = cv2.GaussianBlur(img, (15, 15), 0)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 보라색과 파란색 범위 설정 및 마스크 생성 (HSV 채도, 명도 값 조정)
    # lower_color = np.array([90, 50, 50])  # H value from 100 to include 파랑
    lower_color = np.array([30, 30, 30])  # H value from 100 to include 파랑
    # upper_color = np.array([170, 255, 255])  # H value to 160 to include 보라
    upper_color = np.array([220, 255, 255])  # H value to 160 to include 보라
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 원 검출
    # circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=60,
    #                            param1=50 / 70 / 70 / 50, param2=20 / 10 / 50 / 20, minRadius=130, maxRadius=220)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                               param1=70, param2=40, minRadius=50, maxRadius=190)

    # def remove_duplicate_circles(circles, threshold=40):
    def remove_duplicate_circles(circles, threshold=80):
        unique_circles = []
        for circle in circles:
            x, y, r = circle
            is_unique = True
            for unique_circle in unique_circles:
                unique_x, unique_y, unique_r = unique_circle
                distance = np.sqrt((x - unique_x) ** 2 + (y - unique_y) ** 2)
                if distance < threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_circles.append(circle)
        return np.array(unique_circles)

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        # 중복 원 제거 로직이 있는 경우 아래 줄을 사용하세요.
        unique_circles = remove_duplicate_circles(circles)

        row_count = [0] * 8  # 8개의 행에 대한 카운터

        for i in unique_circles:
            height, width = mask.shape

            # 원 내부에서 색상을 샘플링하여 평균을 구합니다.
            inner_color = np.mean([mask[y, x] for x in range(max(0, i[0] - 5), min(width, i[0] + 6))
                                   for y in range(max(0, i[1] - 5), min(height, i[1] + 6))])

            # 원의 경계에서 색상을 샘플링합니다.
            edge_x, edge_y = int(i[0] + i[2] * 0.7071), int(i[1] + i[2] * 0.7071)  # 대각선 방향으로 이동
            if 0 <= edge_x < width and 0 <= edge_y < height:
                edge_color = mask[edge_y, edge_x]
            else:
                edge_color = 0  # 또는 다른 기본값

            # 원이 속한 행을 찾습니다.
            row = i[1] // (img.shape[0] // 8)

        # 애매한 보라색도 보라색으로 판별
        if inner_color > 50 and abs(inner_color - edge_color) < 50:  # 경계와 중심의 색상이 비슷하면
            row_count[row] += 1
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 255), 8)  # 보라색 원을 굵게 그립니다.
        else:
            cv2.circle(img, (i[0], i[1]), i[2], (255, 255, 255), 8)  # 투명한 원을 굵게 그립니다.

        # 카운트 결과를 이미지 우측에 워터마크로 표시합니다.
        for idx, count in enumerate(row_count):
            watermark_text = f"{count}/12"
            cv2.putText(img, watermark_text, (img.shape[1] - 400, (img.shape[0] // 8) * idx + 300), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 11)

    # 결과 이미지 저장
    output_filepath = os.path.join(current_output_folder, os.path.basename(filepath))
    cv2.imwrite(output_filepath, img)