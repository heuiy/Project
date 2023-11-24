# 튜브 비슷한 애들은 무조건 찾는 듯
# 튜브가 과연 적절한지 고민해봐야 함

# 병합 및 수정된 코드
import cv2
import numpy as np
import os
import glob

# 입력 폴더 경로를 하드코딩합니다.

# input_base_folder = "D:/#.Secure Work Folder/BIG/Project/00Temp/240108_Endo/in/"
# output_base_folder = "D:/#.Secure Work Folder/BIG/Project/00Temp/240108_Endo/out/"

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

def detect_test_tubes(image_path, output_path, folder_name):
    # 이미지를 로드합니다.
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: The image at {image_path} could not be loaded.")
        return 0

    # 이미지의 상단 1/3만 사용합니다.
    height, width = img.shape[:2]
    top_third = img[:height // 3, :]

    # 파란색 검출을 위한 HSV 색상 범위를 설정합니다.
    hsv = cv2.cvtColor(top_third, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])  # 파란색 범위의 하한값을 조정합니다.
    upper_blue = np.array([140, 255, 255])  # 파란색 범위의 상한값을 조정합니다.
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 형태학적 변환을 적용하여 작은 노이즈를 제거합니다.
    kernel = np.ones((5, 5), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

    # 윤곽선 검출
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 파란색 테스트 튜브의 수를 카운트합니다.
    test_tube_count = 0
    for contour in contours:
        # 윤곽선을 둘러싸는 사각형을 구하고, 테스트 튜브로 간주될 형태인지 검사합니다.
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        aspect_ratio = float(w) / h
        if 0.5 < aspect_ratio < 1.5:  # 테스트 튜브 형태의 aspect ratio 기준 설정
            # 여기에 추가적인 형태 분석 및 색상 검증 코드를 삽입합니다.
            # ...

            # 현재는 모든 사각형을 파란색 테스트 튜브로 간주하여 카운트합니다.
            cv2.rectangle(top_third, (x, y), (x + w, y + h), (0, 255, 0), 2)
            test_tube_count += 1

    # 이미지에 검출된 테스트 튜브 개수를 표시합니다.
    cv2.putText(img, f"Endotoxin : {test_tube_count}", (10, top_third.shape[0] - 260),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 2)

    # 이미지에 폴더명을 삽입합니다.
    cv2.putText(img, folder_name, (width - 400, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 2)

    # 결과 이미지를 저장합니다.
    cv2.imwrite(output_path, img)

    return test_tube_count

# 선택된 폴더 내의 모든 jpg 파일에 대해
for filepath in glob.glob(os.path.join(current_input_folder, "*.jpg")):
    # 파일 처리
    output_filepath = os.path.join(current_output_folder, os.path.basename(filepath))
    count = detect_test_tubes(filepath, output_filepath, selected_folder)  # 올바른 함수 이름으로 변경
    print(f"Processed {filepath}: Found {count} test tubes")