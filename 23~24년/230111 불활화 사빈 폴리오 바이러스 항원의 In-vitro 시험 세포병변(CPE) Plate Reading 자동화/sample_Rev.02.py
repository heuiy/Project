# 이미지 상에 x,y 좌표를 250픽셀 간격으로 보여줌

# Sample code to display points at 250-pixel intervals on the image and show their HSV values
import cv2
import os
import glob

def show_points_and_hsv_values(image_path, output_folder, step=250):
    img = cv2.imread(image_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Points will be displayed at an interval defined by 'step'
    for x in range(0, img.shape[1], step):
        for y in range(0, img.shape[0], step):
            # Draw the point
            cv2.circle(img, (x, y), 5, (0, 255, 255), -1)

            # Get the HSV value at this point
            h, s, v = hsv_img[y, x]

            # Display the HSV value next to the point
            cv2.putText(img, f"({h},{s},{v})", (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save the image
    output_image_path = os.path.join(output_folder, "points_and_hsv_values.jpg")
    cv2.imwrite(output_image_path, img)

    print(f"Image with points and HSV values has been saved to {output_image_path}")


# Define the sample and output folders
sample_folder = "D:/#.Secure Work Folder/BIG/Project/00Temp/230829_CPE/in/sample"
output_folder = "D:/#.Secure Work Folder/BIG/Project/00Temp/230829_CPE/out/sample"

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all jpg files in the sample folder
image_files = glob.glob(os.path.join(sample_folder, "*.jpg"))

# Show the available options to the user
print("Please select an image:")
for i, image_path in enumerate(image_files):
    print(f"{i + 1}. {os.path.basename(image_path)}")

# Get the user's choice
selected_index = int(input("Enter the number of the image you want to select: ")) - 1
selected_image_path = image_files[selected_index]

# Show the points and their HSV values on the selected image
show_points_and_hsv_values(selected_image_path, output_folder)
