# Flag!! 중복되지만 그나마 많이 찾았음

# 중복 원 제거 함수가 없어도 원이 하나만 나옴
# 보라색만 깔끔하게 찾아냄. 하지만 10~20% 수준
# 기존에는 보라색 원은 잘 찾았으니까 그걸 초기 모델을 추가

# 여기에 원에 하나씩 번호를 넣고 싶음
# hsv 색깔을 어떻게 정의해야 하는가.....

# 병합 및 수정된 코드
import cv2
import numpy as np
import os
import glob

# 입력 폴더 경로를 하드코딩합니다.

input_base_folder = "D:/#.Secure Work Folder/BIG/Project/00Temp/240108_Endo/in/"
output_base_folder = "D:/#.Secure Work Folder/BIG/Project/00Temp/240108_Endo/out/"

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

def find_similar_circles(image, threshold_value=1):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.medianBlur(gray, 13)

    _, dst = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_TOZERO)

    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, dp=1, minDist=400,
                               param1=40, param2=10, minRadius=150, maxRadius=180)

    valid_circles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles:
            circle_mask = np.zeros_like(image)
            cv2.circle(circle_mask, (x, y), r, (255, 255, 255), -1)
            circle_roi = cv2.bitwise_and(image, circle_mask)

            circle_gray = cv2.cvtColor(circle_roi, cv2.COLOR_BGR2GRAY)
            _, circle_binary = cv2.threshold(circle_gray, 170, 200, cv2.THRESH_TOZERO)

            white_pixels = cv2.countNonZero(circle_binary)

            if white_pixels < 100:
                valid_circles.append((x, y, r))

    return np.array(valid_circles, dtype=np.uint16)

# 선택된 폴더 내의 모든 jpg 파일에 대해
for filepath in glob.glob(os.path.join(current_input_folder, "*.jpg")):
    img = cv2.imread(filepath)

    row_count = [0] * 8  # 8개의 행에 대한 카운터

    unique_circles = find_similar_circles(img)

    for i, (x, y, r) in enumerate(unique_circles):
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