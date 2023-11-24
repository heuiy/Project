# 이미지 파일 경로를 확인하는 스크립트
def check_file_access(image_path):
    try:
        # 파일을 'rb' 모드로 열어서 바이너리로 읽습니다.
        with open(image_path, 'rb') as file:
            print(f"파일을 성공적으로 열었습니다: {image_path}")
            return True
    except Exception as e:
        print(f"파일을 열 수 없습니다: {image_path}")
        print(f"에러: {e}")
        return False

# 사용자의 이미지 파일 경로
image_path = "D:/#.Secure Work Folder/BIG/Project/23~24년/240108 익산QC 엔도톡신 Reading/PIC/in/sample/KakaoTalk_20231106_164652046_10.jpg"

# 파일 접근 확인
check_file_access(image_path)