# 자투리 배치를 마지막에 넣을 경우 (수정 중)

# 예시
# A 첫 번째장 1장 1/10
# B 첫 번째장 3장 2/10 ~ 4/10
# C 첫 번째장 3장 5/10 ~ 7/10
# A 두 번째장 1장 8/10
# B 두 번째장 1장 9/10
# C 두 번째장 1장 10/10

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
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import os
from datetime import datetime


def select_pdf_file():
    download_folder = "C:\\Users\\User\\Downloads\\PDF"
    pdf_files = [f for f in os.listdir(download_folder) if f.endswith('.pdf')]
    print("사용 가능한 PDF 파일:")
    for idx, file in enumerate(pdf_files):
        print(f"{idx + 1}. {file}")
    choice = int(input("선택할 PDF 파일 번호를 입력하세요: ")) - 1
    return os.path.join(download_folder, pdf_files[choice])


def add_watermark(input_pdf_path, type, current_document_number, total_documents):
    positions = {
        'STANDARD_CHNE': (210, 80),
        'STANDARD_CHNF': (248, 44),
        'STANDARD_CHNFA': (394, 326),
        'STANDARD_CHNY': (394, 326),
        'STANDARD_ENG': (248, 44),
        'STANDARD_EUHR': (248, 38),
        'STANDARD_JPNY': (234, 222),
        'STANDARD_SPA': (298, 44),
        'STANDARD_UNICEF': (210, 80)
    }
    reader = PdfFileReader(input_pdf_path)
    writer = PdfFileWriter()
    x_position, y_position = positions[type]
    total_pages = reader.getNumPages()
    for i in range(total_pages):
        page = reader.getPage(i)
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=letter)
        page_number_text = f"{current_document_number + i}              {total_documents}"  # 전체 페이지 수를 출력
        can.drawString(x_position, y_position, page_number_text)
        can.save()
        packet.seek(0)
        watermark = PdfFileReader(packet)
        page.mergePage(watermark.getPage(0))
        writer.addPage(page)
    return writer


def select_type():
    type_options = ['STANDARD_CHNE',
                    'STANDARD_CHNF',
                    'STANDARD_CHNFA',
                    'STANDARD_CHNY',
                    'STANDARD_ENG',
                    'STANDARD_EUHR',
                    'STANDARD_JPNY',
                    'STANDARD_SPA',
                    'STANDARD_UNICEF'
                    ]
    print("유형을 선택하세요:")
    for idx, option in enumerate(type_options):
        print(f"{idx + 1}. {option}")
    type_choice_idx = int(input("유형 번호를 입력하세요: ")) - 1

    if type_choice_idx < 0 or type_choice_idx >= len(type_options):
        raise ValueError("유효하지 않은 유형 번호입니다.")

    return type_options[type_choice_idx]


def main():
    # 전체 페이지 수 입력
    total_documents = int(input("전체 페이지 수를 입력하세요 (예: 10): "))
    current_document_number = 1

    # 배치 페이지 수 입력
    batches = []
    while True:
        first_page_count = input("배치의 첫 번째 페이지 수를 입력하세요 (입력 종료시 'exit' 입력): ")
        if first_page_count.lower() == 'exit':
            break
        second_page_count = int(input("배치의 두 번째 페이지 수를 입력하세요: "))
        batches.append([int(first_page_count), second_page_count])

    # 전체 페이지 수를 고려하여 처리
    for batch_idx, page_counts in enumerate(batches):
        for page_idx, page_count in enumerate(page_counts):
            remaining_count = total_documents - current_document_number + 1
            actual_page_count = min(page_count, remaining_count)

            if actual_page_count <= 0:
                continue

            print(f"배치 {batch_idx + 1} 처리 중, 페이지 {page_idx + 1}")

            print(
                f"배치 {batch_idx + 1}의 페이지 {page_idx + 1}가 처리되었습니다. 페이지 수: {actual_page_count}, 문서 번호: {current_document_number} ~ {current_document_number + actual_page_count - 1}/{total_documents}")

            # 현재 문서 번호 업데이트
            current_document_number += actual_page_count

    print("모든 문서 번호가 추가되었습니다.")

if __name__ == "__main__":
    main()