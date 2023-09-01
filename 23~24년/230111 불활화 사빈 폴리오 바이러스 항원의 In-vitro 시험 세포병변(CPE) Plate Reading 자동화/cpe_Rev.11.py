# 열 이름 4단으로 표기
# 전처리해서 이미지를 800, 600 으로 스케일링

# 폴더명, 파일명에 한글 있으면 안됨
# 큰 거는 아직 분류 못함

# 열 판단 근거
# 유효한 원(valid_circles)을 x 좌표로 정렬한 후, 
# 각 열에 속하는 원의 개수를 세는 column_count 딕셔너리를 사용함. 
# 이 개수를 이미지 상단에 표시함
# x 좌표의 차이가 40보다 크면 새로운 열로 판단하고 있습니다.

# 이미지의 너비(image_width)를 3등분(part_width)하고, 
# 각 칼럼의 인덱스(idx)에 따라 텍스트의 x 좌표(x_position)를 결정합니다. 
# y 좌표(y_position)는 기존과 유사하게 설정되지만, 
# 칼럼 인덱스를 3으로 나눈 나머지를 사용하여 조금씩 아래로 내려갑니다.

# Apply the Hough Circle Transform
# Parameter 설명 출처: https://opencv-python.readthedocs.io/en/latest/doc/26.imageHoughCircleTransform/imageHoughCircleTransform.html
# dp – dp=1이면 Input Image와 동일한 해상도.
# minDist – 검출한 원의 중심과의 최소거리. 값이 작으면 원이 아닌 것들도 검출이 되고, 너무 크면 원을 놓칠 수 있음.
# param1 – 내부적으로 사용하는 canny edge 검출기에 전달되는 Parameter
# param2 – 이 값이 작을 수록 오류가 높아짐. 크면 검출률이 낮아짐.
# minRadius – 원의 최소 반지름.
# maxRadius – 원의 최대 반지름.
# circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
#                             param1=170, param2=10, minRadius=27, maxRadius=32)

# pip install imutils
# pip3 install opencv-python
# pip install matplotlib

import cv2
import numpy as np
import os

def list_subfolders(root_directory):
    folder_dict = {}
    for idx, folder_name in enumerate(os.listdir(root_directory)):
        folder_path = os.path.join(root_directory, folder_name)
        if os.path.isdir(folder_path):
            folder_dict[idx + 1] = folder_name
    return folder_dict

def create_result_folder_and_save_images(original_folder, image_name, processed_image):
    result_folder = f"{original_folder}_result"
    result_folder_path = os.path.join(original_folder, result_folder)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    result_image_path = os.path.join(result_folder_path, f"{image_name}_result.jpg")
    cv2.imwrite(result_image_path, processed_image)

def preprocess_image(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not open or find the image at {image_path}. Skipping...")
        return None

    # 예시: 이미지 스케일링
    standard_size = (800, 600)  # 원하는 표준 크기
    image = cv2.resize(image, standard_size)

    # 이미지 크기 확인
    height, width, _ = image.shape
    print(f"The resized image dimensions are {width}x{height}")

    # 이미지 출력
    cv2.imshow('Resized Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image

def find_similar_circles(image_path, threshold_value=70):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Could not open or find the image at {image_path}. Skipping...")
        return

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 13)
        _, dst = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_TOZERO)
        circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                   param1=170, param2=10, minRadius=27, maxRadius=32)
        
        if circles is None or len(circles) == 0:
            print(f"No circles found in {image_path}. Skipping...")
            return

    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 13)
    _, dst = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_TOZERO)
    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=170, param2=10, minRadius=27, maxRadius=32)
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        valid_circles = []
        for (x, y, r) in circles:
            circle_mask = np.zeros_like(image)
            cv2.circle(circle_mask, (x, y), r, (255, 255, 255), -1)
            circle_roi = cv2.bitwise_and(image, circle_mask)
            circle_gray = cv2.cvtColor(circle_roi, cv2.COLOR_BGR2GRAY)
            _, circle_binary = cv2.threshold(circle_gray,80, 255, cv2.THRESH_TOZERO)
            white_pixels = cv2.countNonZero(circle_binary)
            if white_pixels < 100:
                valid_circles.append((x, y, r))
        circle_count = len(valid_circles)

        # 유효한 원을 x 좌표로 정렬
        valid_circles.sort(key=lambda circle: circle[0])
        
        # 각 열에서 보라색 원의 개수를 세기 위한 딕셔너리
        column_count = {}
        col_idx = 1
        last_x = valid_circles[0][0]
        
        for (x, y, r) in valid_circles:
            if abs(x - last_x) > 40:  # 새로운 열로 고려하기 위한 임계값
                col_idx += 1
            last_x = x
            
            if col_idx not in column_count:
                column_count[col_idx] = 0
            column_count[col_idx] += 1
        
        # 이미지 상단에 각 열의 보라색 원 개수 표시
        image_width = image.shape[1]
        part_width = image_width // 4  # 4등분

        for idx, count in column_count.items():
            if idx <= 2:
                x_position = 20
            elif idx <= 4:
                x_position = part_width + 20
            elif idx <= 6:
                x_position = 2 * part_width + 20
            else:
                x_position = 3 * part_width + 20

            y_position = 50 + (idx - 1) % 2 * 30  # 4등분으로 나누기 때문에 나머지는 0 또는 1

            # 텍스트의 길이를 고려하여 x 좌표를 동적으로 설정
            text = f"Col {idx}: {count}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            adjusted_x_position = x_position - text_size[0] // 2 + 90  # 텍스트의 중심을 x_position으로 맞추고, 오른쪽으로 10 픽셀 이동

            cv2.putText(image, text, (adjusted_x_position, y_position),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)              
        
        for i, (x, y, r) in enumerate(valid_circles):
            cv2.circle(image, (x, y), r, (0, 255, 0), 1)
            cv2.putText(image, str(i+1), (x-3, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
        cv2.putText(image, f"Circle count: {circle_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        save_name = image_path.split('\\')[-1].split('.jpg')[0]
        create_result_folder_and_save_images(os.path.dirname(image_path), save_name, image)

if __name__ == '__main__':
    root_directory = "D:\\#.Secure Work Folder\\BIG\\Project\\00Temp\\230829_CPE\\PIC\\"
    folders = list_subfolders(root_directory)
    print("Select a folder:")
    for idx, folder_name in folders.items():
        print(f"{idx}. {folder_name}")
    user_choice = int(input("Enter the number of your chosen folder: "))
    chosen_folder = folders.get(user_choice, "Invalid choice")
    chosen_folder_path = os.path.join(root_directory, chosen_folder)
    print(f"Selected folder: {chosen_folder}")

    for dirpath, dirnames, filenames in os.walk(chosen_folder_path):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dirpath, filename)
                print(f"Processing {image_path}...")
                try:
                    find_similar_circles(image_path)
                except Exception as e:
                    print(f"An error occurred while processing {image_path}: {e}")
