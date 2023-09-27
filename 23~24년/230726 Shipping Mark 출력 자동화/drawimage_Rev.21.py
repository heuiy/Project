# 자투리 배치 뒤로 모으기
# 각 pdf 서식별로 워터마크 삽입 위치가 없어졌음
# 엄청 간단해졌음

# 2가지 수정사항 반영 중
# 기존 오류
    # 문서 수량이 1장 or 2장일 때 문서 번호를 올바르게 업데이트 하지 못했음
# 수정 사항
    # current_document_number 업데이트 하는 로직 변경함
        # 다음 문서 번호로 업데이트
            # current_document_number += current_batch_copies + (1 if writer.getNumPages() == 2 else 0)
        # Update the current_document_number based on the new total pages
            # current_document_number += new_total_pages

# 출력 pdf 이름 변경
# 전체 페이지 수를 5개 이상의 작은 배치로 계속 나눔

# WMS 와 이름 맞추기
# STANDARD_CHNE             이브아르_중문
# STANDARD_CHNF             폴리트롭_중문
# STANDARD_CHNFA           팩티브_중문
# STANDARD_CHNY             이브아르_Y_SOLUTION_중문
# STANDARD_ENG               표준양식_영문
# STANDARD_EUHR             히루안원_유럽
# STANDARD_JPNY              유셉트_일본
# STANDARD_SPA                표준양식_스페인
# STANDARD_UNICEF           UNICEF

# !pip install pdfrw reportlab
# !pip install PyPDF4

from PyPDF4 import PdfFileReader, PdfFileWriter
from datetime import datetime
import os


def select_pdf_files():
    download_folder = "C:\\Users\\LG\\Downloads"
    pdf_files = [f for f in os.listdir(download_folder) if f.endswith('.pdf')]
    print("사용 가능한 PDF 파일:")
    for idx, file in enumerate(pdf_files):
        print(f"{idx + 1}. {file}")

    selected_indices = input("선택할 PDF 파일 번호들을 입력하세요 (쉼표로 구분): ").split(',')
    selected_files = [os.path.join(download_folder, pdf_files[int(idx) - 1]) for idx in selected_indices]

    return selected_files


def main():
    total_documents = int(input("전체 문서 번호를 입력하세요 (예: 30): "))
    remaining_documents = total_documents  # 남은 문서 수
    second_page_pool = PdfFileWriter()  # 두 번째 페이지를 저장할 변수

    while remaining_documents > 0:
        selected_pdf_files = select_pdf_files()

        for selected_pdf_file in selected_pdf_files:
            input_pdf_filename = os.path.splitext(os.path.basename(selected_pdf_file))[0]
            current_batch_copies = int(input(f"{input_pdf_filename}을(를) 몇 장으로 복사할까요? "))

            # 남은 문서 수보다 크게 입력하면 무시
            if current_batch_copies > remaining_documents:
                print("입력한 배치 페이지 수가 남은 문서 수보다 클 수 없습니다. 다시 시도해주세요.")
                continue

            reader = PdfFileReader(selected_pdf_file)
            total_pages = reader.getNumPages()

            # 첫 번째 페이지 복사
            writer = PdfFileWriter()
            for _ in range(current_batch_copies):
                writer.addPage(reader.getPage(0))

            # 임시 파일에 저장
            temp_pdf_file = f"C:\\Users\\LG\\Downloads\\TEMP\\temp_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pdf"
            with open(temp_pdf_file, "wb") as temp_pdf:
                writer.write(temp_pdf)

            # 두 번째 페이지가 있다면 저장
            if total_pages == 2:
                second_page_pool.addPage(reader.getPage(1))

            # 출력 파일 생성
            output_pdf_file = os.path.join("C:\\Users\\LG\\Downloads\\OUT",
                                           f"{input_pdf_filename}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pdf")
            with open(output_pdf_file, "wb") as output_pdf:
                writer.write(output_pdf)

            remaining_documents -= current_batch_copies  # 남은 문서 수 업데이트

    # 두 번째 페이지를 모아서 마지막에 출력
    if second_page_pool.getNumPages() > 0:
        output_pdf_file_second_page = os.path.join("C:\\Users\\LG\\Downloads\\OUT",
                                                   f"all_second_pages_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pdf")
        with open(output_pdf_file_second_page, "wb") as output_pdf:
            second_page_pool.write(output_pdf)

    print("모든 문서 번호가 추가되었습니다.")

if __name__ == "__main__":
    main()