# 보라색 원 찾기
# HSV 색공간에서 보라색 범위에 속하는 픽셀만을 대상으로 원을 찾음
# 원 내부에서 여러 픽셀의 색상을 샘플링하여 평균을 구함

# 색상 범위 확장
# 파란색부터 보라색, 그리고 연보라색까지 포함하도록
    # HSV 값의 범위를 확장

# 애매한 색상 판별:
# 보라색과 투명색을 더 정확하게 구분하려면,
# 각 원의 중심에서 색상을 샘플링하는 대신
# 원 내부의 여러 픽셀을 샘플링하여 평균 색상을 구함.
# 이 평균 색상을 사용하여 보라색을 판별
    # 문제 : 경계선만 보라색이어도 검출됨

# 투명한 원 판별:
# 원의 경계선만 보라색인 경우를 판별하기 위해
# 원의 중심과 경계에서 색상을 샘플링
# 두 색상이 크게 다르면 투명한 원으로 판별

# 중복 원 제거:
# unique_circles 배열을 사용하여 중복 원 제거
    # 검출된 원들 사이의 거리를 계산하여 가까운 원을 제거
# 여전히 작은 원과 큰 원이 중복으로 검출
# 거리 뿐만 아니라 반지름도 고려하여 중복 원을 제거함

# cv2.GaussianBlur 함수
    # 이미지에 가우시안 블러(Gaussian Blur)를 적용하는 OpenCV의 함수
    # 가우시안 블러는 이미지의 노이즈를 줄이고
    # 세부 사항을 흐리게 하는 데 사용됨.
    # 주로 에지(edge) 검출이나 다른 이미지 처리 작업을 수행하기 전에 전처리 단계로 사용됨
        # img: 블러를 적용할 원본 이미지입니다.
        # (15, 15): 가우시안 커널의 크기. 이 값이 크면 블러 효과가 더 강하게 적용됩니다. 일반적으로 양의 홀수를 사용합니다.
            # 0: 가우시안 커널의 X 방향 표준편차입니다. 이 값이 0이면 (15, 15) 크기의 커널에서 자동으로 계산됩니다.
            # 즉, cv2.GaussianBlur(img, (15, 15), 0) 코드는 원본 이미지 img에 크기 (15, 15)의 가우시안 블러를 적용하고 그 결과를 blurred에 저장함.
            # 이 블러 처리는 원본 이미지의 노이즈를 줄이는 데 도움을 줄 수 있으며,
            # 이후의 이미지 처리 작업을 더 안정적으로 만들어 줄 수 있음

# lower_color = np.array([90, 50, 50])  # H value from 100 to include 파랑
# upper_color = np.array([170, 255, 255])  # H value to 160 to include 보라

# upper_color = np.array([170, 255 | 100, 255 | 100])  # H value to 160 to include 보라

    # HSV (Hue, Saturation, Value) 색 공간에서 보라색과 파란색 범위를 설정하고,
        # 해당 범위에 속하는 픽셀만을 대상으로 마스크를 생성합니다.

    # lower_color = np.array([90, 50, 50]):
        # 이 배열은 HSV 색 공간에서의 색상 범위의 하한을 설정합니다.
    # Hue (H) 값이 90:
        # 이 값은 파란색과 보라색의 하한을 설정합니다.
        # Hue 값은 일반적으로 0~179 범위로 표현됩니다.
    # Saturation (S) 값이 50:
        # 이 값은 색상의 채도를 설정합니다.
        # 값이 낮을수록 흰색에 가깝고, 값이 높을수록 순수한 색상에 가까워집니다.
    # Value (V) 값이 50:
        # 이 값은 색상의 명도를 설정합니다.
        # 값이 낮을수록 검은색에 가깝고, 값이 높을수록 원래 색상에 가까워집니다.

    # upper_color = np.array([170, 255, 255]): 이 배열은 HSV 색 공간에서의 색상 범위의 상한을 설정합니다.
        # Hue (H) 값이 170: 이 값은 보라색의 상한을 설정합니다.
        # Saturation (S) 값이 255: 채도의 최대값을 의미합니다.
        # Value (V) 값이 255: 명도의 최대값을 의미합니다.
    # 이렇게 설정된 lower_color와 upper_color는 cv2.inRange(hsv, lower_color, upper_color) 함수에서 사용되어,
        # 지정된 범위에 속하는 픽셀은 흰색 (255)으로,
        # 그렇지 않은 픽셀은 검은색 (0)으로 설정되는 이진 마스크 이미지를 생성합니다.
        # 이 마스크 이미지는 이후의 원 검출 등의 작업에 사용됩니다.

# circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=30 | 60
#                            param1=50, param2=30, minRadius=10 | 130 , maxRadius=50 | 220)
    # method: 검출 방법 (일반적으로 cv2.HOUGH_GRADIENT를 사용)
    # dp: 이미지 해상도에 대한 축소 비율 (1이면 원래 크기)
    # minDist: 60으로 설정하여 원 간의 최소 거리를 늘렸습니다.
        # 이 값을 줄여야 하는 경우. 원이 너무 가까이 있어서 하나로 간주하려면 이 값을 줄이면 됨
        # 이 값을 늘려야 하는 경우. 원이 너무 멀리 떨어져 있으면 이 값을 늘려야 합니다.
    # param1 – 내부적으로 사용하는 canny edge 검출기에 전달되는 Parameter
    # param2 – 두 번째 메서드 파라미터
        # 이 값이 작을 수록 오류가 높아짐. 더 많은 원이 검출됨
        # 이 값이 크면 검출률이 낮아짐. 더 적은 원이 검출됨
        # 허프 원 변환에서 원의 탐지 품질 조정이 가능함
    # param1 = 50, param2 = 40:
        # 이 값들은 각각 Canny edge 검출기와 검출된 원의 중심을 필터링하는데 사용됨.
    # minRadius: 이 값보다 작은 반지름을 가진 원은 무시됩니다.
    # maxRadius: 이 값보다 큰 반지름을 가진 원은 무시됩니다.

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
    #                            param1=50 / 70 / 70, param2=20 / 10 / 50, minRadius=130, maxRadius=220)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=60,
                               param1=50, param2=20, minRadius=130, maxRadius=220)

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        row_count = [0]*8  # 8개의 행에 대한 카운터

        # 원 중복 제거
        unique_circles = []
        for circle in circles:
            is_unique = True
            for unique_circle in unique_circles:
                distance = np.linalg.norm(circle[:2] - unique_circle[:2])
                radius_diff = abs(int(circle[2]) - int(unique_circle[2]))  # 데이터 타입을 int로 변환

                # if distance < 30 and radius_diff < 20:    20 / 5 , 30 / 15
                if distance < 30 and radius_diff < 20:
                    is_unique = False
                    break

            if is_unique:
                unique_circles.append(circle)

        for i in unique_circles:
            # 원 내부에서 여러 픽셀의 색상을 샘플링하여 평균을 구합니다.
            height, width = mask.shape
            inner_color = np.mean([mask[y, x] for x in range(max(0, i[0] - 5), min(width, i[0] + 6)) for y in
                                   range(max(0, i[1] - 5), min(height, i[1] + 6))])

            # 원의 경계에서 색상을 샘플링합니다.
            edge_x, edge_y = int(i[0] + i[2] * 0.7071), int(i[1] + i[2] * 0.7071)  # 대략적으로 원의 경계
            height, width = mask.shape

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
            cv2.putText(img, watermark_text, (img.shape[1] - 300, (img.shape[0] // 8) * idx + 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 9)

    # 결과 이미지 저장
    output_filepath = os.path.join(current_output_folder, os.path.basename(filepath))
    cv2.imwrite(output_filepath, img)