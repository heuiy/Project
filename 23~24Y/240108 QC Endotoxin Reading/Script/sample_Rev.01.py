import cv2
import os
import glob

# URL 인코딩으로 경로를 안전하게 만듭니다.
sample_folder = "D:/#.Secure Work Folder/BIG/Project/23~24년/240108 익산QC 엔도톡신 Reading/PIC/in".replace("#", "%23")
output_folder = "D:/#.Secure Work Folder/BIG/Project/23~24년/240108 익산QC 엔도톡신 Reading/PIC/out".replace("#", "%23")

# 출력 폴더가 없으면 생성합니다.
os.makedirs(output_folder, exist_ok=True)

# 폴더 내의 모든 jpg 파일을 찾습니다.
image_files = glob.glob(os.path.join(sample_folder, "*.jpg"))

# 사용자에게 선택할 수 있는 옵션을 제공합니다.
print("Please select an image:")
for i, image_path in enumerate(image_files):
    print(f"{i + 1}. {os.path.basename(image_path)}")

# 사용자가 선택한 옵션을 가져옵니다.
selected_index = int(input("Enter the number of the image you want to select: ")) - 1
selected_image_path = image_files[selected_index]

# 선택한 이미지를 읽어옵니다.
img = cv2.imread(selected_image_path)

# 이미지가 정상적으로 로드되었는지 확인합니다.
if img is None:
    print(f"Cannot load the image: {selected_image_path}")
else:
    # 텍스트를 표시할 간격을 설정합니다.
    step = 250

    # x, y 좌표를 표시합니다.
    for x in range(0, img.shape[1], step):
        for y in range(0, img.shape[0], step):
            cv2.putText(img, f"({x},{y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 결과 이미지를 저장합니다.
    output_image_path = os.path.join(output_folder, "coordinates.jpg")
    cv2.imwrite(output_image_path, img)

    print(f"Image with coordinates has been saved to {output_image_path}")

# 텍스트를 표시할 간격을 설정합니다.
step = 250

# x, y 좌표를 표시합니다.
for x in range(0, img.shape[1], step):
    for y in range(0, img.shape[0], step):
        cv2.putText(img, f"({x},{y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# 결과 이미지를 저장합니다.
output_image_path = os.path.join(output_folder, "coordinates.jpg")
cv2.imwrite(output_image_path, img)

print(f"Image with coordinates has been saved to {output_image_path}")
