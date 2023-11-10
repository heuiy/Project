# 좀 불완전한데 그래도 전처리 코드를 넣으려고 했음

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
import imutils  # 스캔 이미지
import matplotlib.pyplot as plt

def list_subfolders(root_directory):
    folder_dict = {}
    for idx, folder_name in enumerate(os.listdir(root_directory)):
        folder_path = os.path.join(root_directory, folder_name)
        if os.path.isdir(folder_path):
            folder_dict[idx + 1] = folder_name
    return folder_dict

def create_result_folder_and_save_images(original_folder, image_name, processed_image):
    result_folder = f"{original_folder.split(os.sep)[-1]}_result"  # 폴더 이름만 추출해서 _result를 붙임
    result_folder_path = os.path.join(os.path.dirname(original_folder), result_folder)  # 원래 폴더와 같은 위치에 _result 폴더 생성
    
    print(f"Debug: result_folder_path = {result_folder_path}")  # 디버깅용 출력
    
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    
    result_image_path = os.path.join(result_folder_path, f"{image_name}_result.jpg")
    
    print(f"Debug: result_image_path = {result_image_path}")  # 디버깅용 출력
    
    cv2.imwrite(result_image_path, processed_image)

#def create_result_folder_and_save_images(original_folder, image_name, processed_image):
    #result_folder = f"{original_folder.split(os.sep)[-1]}_result"  # 폴더 이름만 추출해서 _result를 붙임
    #result_folder_path = os.path.join(os.path.dirname(original_folder), result_folder)  # 원래 폴더와 같은 위치에 _result 폴더 생성
    #if not os.path.exists(result_folder_path):
    #    os.makedirs(result_folder_path)
    #result_image_path = os.path.join(result_folder_path, f"{image_name}_result.jpg")
    #cv2.imwrite(result_image_path, processed_image)

# 스캔한 문서처럼 이미지를 만들어주는 함수
def make_scan_image(image, width, ksize=(5,5), min_threshold=75, max_threshold=200):
    image_list_title = []
    image_list = []

    org_image = image.copy()
    image = imutils.resize(image, width=width)
    ratio = org_image.shape[1] / float(image.shape[1])

    # 이미지를 grayscale로 변환하고 blur를 적용
    # 모서리를 찾기위한 이미지 연산
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, ksize, 0)
    edged = cv2.Canny(blurred, min_threshold, max_threshold)

    image_list_title = ['gray', 'blurred', 'edged']
    image_list = [gray, blurred, edged]

    # contours를 찾아 크기순으로 정렬
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    findCnt = None

    # Contour란 같은 값을 가진 곳을 연결한 선. 이미지의 외곽선을 검출하기 위해 사용.
    # ```cv2.findContours(image, mode, method, contours=None, hierarchy=None, offset=None) -> contours```
    # • image: 입력 이미지. non-zero 픽셀을 객체로 간주함  
    # • mode: 외곽선 검출 모드. cv2.RETR_로 시작하는 상수  
    # • method: 외곽선 근사화 방법. cv2.CHAIN_APPROX_로 시작하는 상수

    # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
        if len(approx) == 4:
            findCnt = approx
            break

    # 만약 추출한 윤곽이 없을 경우 오류
    if findCnt is None:
        raise Exception(("Could not find outline."))

    output = image.copy()
    cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)

    image_list_title.append("Outline")
    image_list.append(output)

    # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
    transform_image = four_point_transform(org_image, findCnt.reshape(4, 2) * ratio)

    # Matplotlib 사용
    plt.imshow(cv2.cvtColor(transform_image, cv2.COLOR_BGR2RGB))
    plt.title("Transform")
    plt.show()

    return transform_image

def four_point_transform(image, pts):
    rect = np.array([
        [pts[0][0], pts[0][1]],
        [pts[1][0], pts[1][1]],
        [pts[2][0], pts[2][1]],
        [pts[3][0], pts[3][1]]
    ], dtype="float32")
  
    (tl, tr, br, bl) = rect
  
    widthA = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    widthB = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    maxWidth = max(int(widthA), int(widthB))
  
    heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    maxHeight = max(int(heightA), int(heightB))
  
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
  
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
  
    return warped

def find_similar_circles(image_path, threshold_value=70):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Could not open or find the image at {image_path}. Skipping...")
        return

    # Check if the image is in landscape mode and rotate if necessary
    if image.shape[1] > image.shape[0]:  # if width > height
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

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

    # 이미지를 스캔한 문서처럼 만들기
    try:
        image = make_scan_image(image, width=500)  # 필요에 따라 너비를 수정할 수 있습니다
    except Exception as e:
        print(f"{image_path}를 스캔하는 동안 오류가 발생했습니다: {e}")
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
