import cv2

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Threshold Demo'

def Threshold_Demo(val):
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
    _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, threshold_type)
    cv2.imshow(window_name, dst)

if __name__ == '__main__':
    src = cv2.imread("D:/#.Secure Work Folder/230225_CPE-20230313T010318Z-001/230225_CPE/KakaoTalk_20230220_171652950_14.jpg")

    # Convert the image to Gray
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # src_gray = cv.medianBlur(src_gray,15)
    cv2.namedWindow(window_name)
    cv2.createTrackbar(trackbar_type, window_name, 3, max_type, lambda x: None)
    cv2.createTrackbar(trackbar_value, window_name, 0, max_value, Threshold_Demo)
    Threshold_Demo(0)
    cv2.waitKey()
