# 폴더명, 파일명에 한글 있으면 안됨
# 큰 거는 아직 분류 못함

# 열 판단 근거
# 유효한 원(valid_circles)을 x 좌표로 정렬한 후, 
# 각 열에 속하는 원의 개수를 세는 column_count 딕셔너리를 사용함. 
# 이 개수를 이미지 상단에 표시함
# x 좌표의 차이가 40보다 크면 새로운 열로 판단하고 있습니다.

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

def find_similar_circles(image_path, threshold_value=70):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Could not open or find the image at {image_path}. Skipping...")
        return

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
