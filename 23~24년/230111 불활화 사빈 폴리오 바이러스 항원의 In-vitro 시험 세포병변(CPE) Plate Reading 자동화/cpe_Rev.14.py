import cv2
import numpy as np
import os
import glob

def list_subfolders(root_directory):
    folder_dict = {}
    for idx, folder_name in enumerate(os.listdir(root_directory)):
        folder_path = os.path.join(root_directory, folder_name)
        if os.path.isdir(folder_path):
            folder_dict[idx + 1] = folder_name
    return folder_dict

def find_similar_circles(image_path, output_directory):
    img = cv2.imread(image_path)
    # ... (이미지 처리 코드는 동일합니다.)

# 루트 디렉터리 설정
root_directory = "D:/#.Secure Work Folder/BIG/Project/00Temp/230829_CPE/in/"

# 사용자에게 폴더 리스트를 보여주고 선택하게 합니다.
folders = list_subfolders(root_directory)
print("Select a folder:")
for idx, folder_name in folders.items():
    print(f"{idx}. {folder_name}")

user_choice = int(input("Enter the number of your chosen folder: "))
chosen_folder = folders.get(user_choice, "Invalid choice")

if chosen_folder == "Invalid choice":
    print("Invalid choice. Exiting.")
    exit(1)

input_folder = os.path.join(root_directory, chosen_folder)
output_folder = f"D:/#.Secure Work Folder/BIG/Project/00Temp/230829_CPE/out/{chosen_folder}"

# 폴더 안의 모든 jpg 파일에 대해
for filepath in glob.glob(os.path.join(input_folder, "*.jpg")):
    print(f"Processing {filepath}...")
    try:
        find_similar_circles(filepath, output_folder)
    except Exception as e:
        print(f"An error occurred while processing {filepath}: {e}")

