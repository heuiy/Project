import cv2
import numpy as np

def find_similar_circles(image_path, threshold_value=70):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.medianBlur(gray, 13)

    _, dst = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_TOZERO)

    # Apply the Hough Circle Transform
    # Parameter 설명 출처: https://opencv-python.readthedocs.io/en/latest/doc/26.imageHoughCircleTransform/imageHoughCircleTransform.html
    # dp – dp=1이면 Input Image와 동일한 해상도.
    # minDist – 검출한 원의 중심과의 최소거리. 값이 작으면 원이 아닌 것들도 검출이 되고, 너무 크면 원을 놓칠 수 있음.
    # param1 – 내부적으로 사용하는 canny edge 검출기에 전달되는 Parameter
    # param2 – 이 값이 작을 수록 오류가 높아짐. 크면 검출률이 낮아짐.
    # minRadius – 원의 최소 반지름.
    # maxRadius – 원의 최대 반지름.
    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=170, param2=10, minRadius=27, maxRadius=32)

    if circles is not None:
        # Round the circle parameters to integers
        circles = np.round(circles[0, :]).astype(int)

        # Initialize a list to store valid circles
        valid_circles = []

        for (x, y, r) in circles:
            # Extract the circle region from the original image
            circle_mask = np.zeros_like(image)
            cv2.circle(circle_mask, (x, y), r, (255, 255, 255), -1)
            circle_roi = cv2.bitwise_and(image, circle_mask)

            # Convert the circle region to grayscale
            circle_gray = cv2.cvtColor(circle_roi, cv2.COLOR_BGR2GRAY)

            # Apply thresholding to obtain a binary mask
            _, circle_binary = cv2.threshold(circle_gray,80, 255, cv2.THRESH_TOZERO)

            # Count the number of white pixels in each circle
            white_pixels = cv2.countNonZero(circle_binary)

            print((x, y, r), white_pixels)
            if white_pixels < 100:
                valid_circles.append((x, y, r))

        # Count the number of valid circles
        circle_count = len(valid_circles)

        for i, (x, y, r) in enumerate(valid_circles):
            cv2.circle(image, (x, y), r, (0, 255, 0), 1)
            cv2.putText(image, str(i+1), (x-3, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)

        cv2.putText(image, f"Circle count: {circle_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display the image with circles
        cv2.namedWindow('Final_result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Final_result', width=900, height=1100)
        cv2.imshow("Final_result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Save the image
        save_name = image_path.split('/')[-1].split('.jpg')[0]
        save_name = save_name + '_result.jpg'
        cv2.imwrite(save_name, image)


if __name__ == '__main__':
    # Load the image
    image_path = "D:/#.Secure Work Folder/230225_CPE-20230313T010318Z-001/230225_CPE/KakaoTalk_20230220_171652950_13.jpg"

    # Find and display circles in the image
    # parameter: image_path, threshold_value=70
    try:
        find_similar_circles(image_path)
    except Exception as e: print(e)
