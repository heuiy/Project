from pdf_file_manager import select_pdf_file, copy_pages
from watermark_manager import add_watermark
from type_selector import select_type
import os
from datetime import datetime

def main():
    download_folder = "C:\\Users\\LG\\Downloads"
    total_documents = int(input("전체 문서 번호를 입력하세요 (예: 30): "))
    first_pdf_copies = int(input("첫 번째 출력되는 PDF 페이지 수를 입력하세요 (예: 20): "))
    
    if first_pdf_copies > total_documents:
        print("첫 번째 PDF 페이지 수는 전체 문서 번호보다 클 수 없습니다. 다시 시도해주세요.")
        return

    current_document_number = 1

    # 첫 번째 PDF 파일 처리
    selected_pdf_file = select_pdf_file(download_folder) # download_folder를 전달
    writer = copy_pages(selected_pdf_file, first_pdf_copies)
    if writer is None:
        print("PDF 생성에 실패하였습니다.")
    else:
        temp_pdf_file = os.path.join("C:\\Users\\LG\\Downloads", "temp.pdf")
        with open(temp_pdf_file, "wb") as temp_pdf:
            writer.write(temp_pdf)

        type_choice = select_type()

        # temp.pdf 파일에 워터마크 추가
        writer_with_watermark = add_watermark(temp_pdf_file, type_choice, current_document_number, total_documents)

        # 파일 이름에 current_document_number 추가
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        output_pdf_file = os.path.join("C:\\Users\\LG\\Downloads", f"{timestamp}_{current_document_number}.pdf")
        with open(output_pdf_file, "wb") as output_pdf:
            writer_with_watermark.write(output_pdf)

        print(f"페이지 번호가 추가된 파일이 저장되었습니다: {output_pdf_file}")

        # 다음 문서 번호로 업데이트
        current_document_number += first_pdf_copies + (1 if writer.getNumPages() == 2 else 0)

    # 두 번째 PDF 파일 처리 (첫 번째 PDF 파일이 전체 문서 번호를 채우지 않았을 경우)
    if current_document_number <= total_documents:
        second_pdf_copies = total_documents - current_document_number + 1
        selected_pdf_file = select_pdf_file(download_folder) # download_folder를 전달
        writer = copy_pages(selected_pdf_file, second_pdf_copies)

        # 두 번째 PDF 파일에 대한 나머지 처리
        temp_pdf_file = os.path.join("C:\\Users\\LG\\Downloads", "temp.pdf")
        with open(temp_pdf_file, "wb") as temp_pdf:
            writer.write(temp_pdf)

        type_choice = select_type()

        # temp.pdf 파일에 워터마크 추가
        writer_with_watermark = add_watermark(temp_pdf_file, type_choice, current_document_number, total_documents)
        
        # 파일 이름에 current_document_number 추가
        output_pdf_file = os.path.join("C:\\Users\\LG\\Downloads", f"{timestamp}_{current_document_number}.pdf")
        with open(output_pdf_file, "wb") as output_pdf:
            writer_with_watermark.write(output_pdf)

        print(f"페이지 번호가 추가된 파일이 저장되었습니다: {output_pdf_file}")

    print("모든 문서 번호가 추가되었습니다.")

if __name__ == "__main__":
    main()