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
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough Transform을 사용하여 원 검출
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

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
roi_coordinates = (100, 100, 200, 200)

# 원의 반지름을 계산
radius = get_circle_radius(image_path, roi_coordinates)
if radius:
    print(f"Estimated average circle radius in the ROI: {radius}")
else:
    print("No circles found in the ROI.")
