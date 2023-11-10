import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread('D:/#.Secure Work Folder/BIG/Project/00Temp/230829_CPE/in/CBA28005/IMG_0203.jpg')

# 이미지 전처리 (노이즈 제거)
blurred = cv2.GaussianBlur(img, (15, 15), 0)

# HSV 색공간으로 변환
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# 보라색 범위 설정
lower_purple = np.array([130, 50, 50])
upper_purple = np.array([160, 255, 255])

# 보라색 마스크 생성
mask = cv2.inRange(hsv, lower_purple, upper_purple)

# 원 검출
circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                          param1=50, param2=30, minRadius=10, maxRadius=50)

# 원이 검출되었다면
if circles is not None:
   circles = np.uint16(np.around(circles))
   count = 0
   for i in circles[0, :]:
       # 원 그리기
       cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
       count += 1

   print(f"Total purple circles: {count}")

# 결과 이미지 보기
cv2.imshow('Detected Circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

