import cv2
import numpy as np


def get_circle_radius(image_path, roi_coordinates):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not open or find the image at {image_path}.")
        return None

    x, y, w, h = roi_coordinates
    roi = image[y:y + h, x:x + w]

    # ROI를 이미지에 그리기
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ROI에 대한 이미지 출력
    cv2.imshow('Image with ROI', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 이미지 전처리: 블러링 적용
    blurred_gray = cv2.GaussianBlur(gray, (3, 3), 2, 2)

    # Hough Transform을 사용하여 원 검출: 파라미터 수정
    circles = cv2.HoughCircles(blurred_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                               param1=50, param2=20, minRadius=5, maxRadius=50)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        radii = [r for (_, _, r) in circles]
        avg_radius = np.mean(radii)
        return avg_radius
    else:
        return None

# 이미지 경로 설정
image_path = "D:\\#.Secure Work Folder\\BIG\\Project\\00Temp\\230829_CPE\\in\\ABC23001\\IMG_0055.JPG"

# ROI 좌표를 설정 (x, y, width, height)
roi_coordinates = (1000, 1000, 200, 200)

# 원의 반지름을 계산
radius = get_circle_radius(image_path, roi_coordinates)
if radius:
    print(f"Estimated average circle radius in the ROI: {radius}")
else:
    print("No circles found in the ROI.")
