# 배치 번호를 이미지에 삽입함
# 파란색을 전혀 못 찾고 있음

# 병합 및 수정된 코드
import cv2
import numpy as np
import os
import glob

# 입력 폴더 경로를 하드코딩합니다.
# input_base_folder = "D:/#.Secure Work Folder/BIG/Project/00Temp/240108_Endo/in/"
# output_base_folder = "D:/#.Secure Work Folder/BIG/Project/00Temp/240108_Endo/out/"

# input_base_folder = "D:/#.Secure Work Folder/BIG/Project/23_24년/240108QCEndotoxinReading/in/"
# output_base_folder = "D:/#.Secure Work Folder/BIG/Project/23_24년/240108QCEndotoxinReading/out/"
#
# input_base_folder = "D:/#.Secure Work Folder/BIG/Project/23_24년/240108QCEndotoxinReading/PIC/in/"
# output_base_folder = "D:/#.Secure Work Folder/BIG/Project/23_24년/240108QCEndotoxinReading/PIC/out/"

input_base_folder = "D:/#.Secure Work Folder/BIG/Project/23~24Y/240108 QC Endotoxin Reading/PIC/in/"
output_base_folder = "D:/#.Secure Work Folder/BIG/Project/23~24Y/240108 QC Endotoxin Reading/PIC/out/"

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

def detect_blue_semicircles(image_path, output_path, folder_name):
    # 이미지를 로드합니다.
    img = cv2.imread(image_path)
    if img is None:  # 이미지가 제대로 로드되지 않았는지 확인합니다.
        print(f"Error: The image at {image_path} could not be loaded.")
        return 0

    # 이미지의 상단 1/3만 사용합니다.
    img_top_third = img[:img.shape[0]//3, :]  # 이미지의 상단 1/3 영역을 자릅니다.

    # 파란색을 검출하기 위한 HSV 색상 범위를 설정합니다.
    # HSV 색상 범위는 실험을 통해 조정할 수 있습니다.
    hsv = cv2.cvtColor(img_top_third, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50])  # 파란색 범위의 하한값 조정
    upper_blue = np.array([140, 255, 255])  # 파란색 범위의 상한값 조정
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Hough 변환 파라미터 조정
    circles = cv2.HoughCircles(mask_blue, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                               param1=20, param2=10, minRadius=5, maxRadius=10)

    # 파란색 반원의 수를 카운트합니다.
    blue_semicircle_count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])  # 원의 중심을 구합니다.
            radius = i[2]  # 원의 반지름을 구합니다.
            # 반원 영역의 파란색 픽셀 수를 검사합니다.
            mask_for_circle = np.zeros_like(mask_blue)
            cv2.circle(mask_for_circle, center, radius, 255, thickness=-1)
            blue_area_in_circle = cv2.bitwise_and(mask_blue, mask_blue, mask=mask_for_circle)
            top_semicircle_area = blue_area_in_circle[:radius, :]  # 반원의 상단부분을 잘라냅니다.
            if np.count_nonzero(top_semicircle_area) > radius * 10:  # 일정 픽셀 이상이면 파란색으로 간주합니다.
                cv2.circle(img_top_third, center, radius, (255, 0, 0), 2)  # 이미지에 원을 그립니다.
                blue_semicircle_count += 1

    # 원본 이미지에 폴더명 삽입
    cv2.putText(img, folder_name, (img.shape[1] - 300, img.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 2)

    # 원본 이미지 크기에 상단 1/3에 표시한 결과를 저장합니다.
    cv2.imwrite(output_path, img)

# 선택된 폴더 내의 모든 jpg 파일에 대해
for filepath in glob.glob(os.path.join(current_input_folder, "*.jpg")):
    # 파일 처리
    output_filepath = os.path.join(current_output_folder, os.path.basename(filepath))
    count = detect_blue_semicircles(filepath, output_filepath, selected_folder)  # 폴더명을 인자로 추가
    print(f"Processed {filepath}: Found {count} blue semicircles")
