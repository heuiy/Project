# 오류는 안 나는 조건
# 대신 하나도 못 찾고 있음

# 엄청 Detect 됨

# 여기에 원에 하나씩 번호를 넣고 싶음

# 병합 및 수정된 코드
import cv2
import numpy as np
import os
import glob

# def remove_duplicate_circles(circles, threshold=150): # 150 이면 너무 커서 중복 원이 거의 제거 안됨. 원 엄청 많아짐!!
def remove_duplicate_circles(circles, threshold=1):
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

# 입력 폴더 경로를 하드코딩합니다.
input_base_folder = "D:/#.Secure Work Folder/BIG/Project/00Temp/230829_CPE/in/"
output_base_folder = "D:/#.Secure Work Folder/BIG/Project/00Temp/230829_CPE/out/"

# 입력 폴더 내의 하위 폴더 목록을 출력하고 사용자에게 선택하게 합니다.
subfolders = [f.name for f in os.scandir(input_base_folder) if f.is_dir()]
for i, folder_name in enumerate(subfolders):
    print(f"{i + 1}. {folder_name}")

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

    row_count = [0] * 8  # 8개의 행에 대한 카운터

    # 이미지 전처리
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 보라색과 파란색 범위 설정 및 마스크 생성 (HSV 채도, 명도 값 조정)
    # lower_color = np.array([30/80, 30/50, 30/50])
    # upper_color = np.array([220/160, 255, 255])
    lower_color = np.array([100, 150, 50])
    upper_color = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 원 검출
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=60,
                               param1=50, param2=40, minRadius=130, maxRadius=180)

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        unique_circles = remove_duplicate_circles(circles)

        for i, (x, y, r) in enumerate(unique_circles):
            # 원이 속한 행을 찾습니다.
            row = y // (img.shape[0] // 8)
            row_count[row] += 1  # 해당 행의 카운트 증가

            cv2.circle(img, (x, y), r, (0, 255, 0), 5)
            cv2.putText(img, str(i + 1), (x - 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 10)

    # 카운트 결과를 이미지 우측에 워터마크로 표시합니다.
    for idx, count in enumerate(row_count):
        watermark_text = f"{count}/12"
        cv2.putText(img, watermark_text, (img.shape[1] - 400, (img.shape[0] // 8) * idx + 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 11)

    # 결과 이미지 저장
    output_filepath = os.path.join(current_output_folder, os.path.basename(filepath))
    cv2.imwrite(output_filepath, img)

