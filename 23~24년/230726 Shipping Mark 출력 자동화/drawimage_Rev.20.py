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
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import os
from datetime import datetime

def select_pdf_file():
    download_folder = "C:\\Users\\LG\\Downloads"
    # download_folder = "C:\\Users\\User\\Downloads\\PDF"
    pdf_files = [f for f in os.listdir(download_folder) if f.endswith('.pdf')]
    print("사용 가능한 PDF 파일:")
    for idx, file in enumerate(pdf_files):
        print(f"{idx + 1}. {file}")
    choice = int(input("선택할 PDF 파일 번호를 입력하세요: ")) - 1
    return os.path.join(download_folder, pdf_files[choice])

def copy_pages(input_pdf_path, first_copies):
    reader = PdfFileReader(input_pdf_path)
    writer = PdfFileWriter()
    total_pages = reader.getNumPages()

    if total_pages not in [1, 2]:
        print("지원하지 않는 페이지 수입니다. 1페이지 또는 2페이지의 PDF만 지원됩니다.")
        return None

    if total_pages == 2: # 두 번째 페이지가 있으면 첫 번째 페이지 복사 수를 하나 줄임
        first_copies -= 1

    for _ in range(first_copies):
        writer.addPage(reader.getPage(0))
    if total_pages == 2: # 두 번째 페이지가 있을 때 한 장만 추가
        writer.addPage(reader.getPage(1))

    return writer

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
    total_documents = int(input("전체 문서 번호를 입력하세요 (예: 30): "))
    current_document_number = 1

    while current_document_number <= total_documents:
        # 남은 문서 수 계산
        remaining_documents = total_documents - current_document_number + 1
        
        print(f"남은 문서 수: {remaining_documents}")
        
        # 현재 배치에 복사할 페이지 수 입력
        current_batch_copies = int(input(f"{current_document_number}부터 시작하는 현재 배치를 몇 장으로 할까요? "))
        
        if current_batch_copies > remaining_documents:
            print("입력한 배치 페이지 수가 남은 문서 수보다 클 수 없습니다. 다시 시도해주세요.")
            continue

        # PDF 파일 선택 및 복사
        selected_pdf_file = select_pdf_file()
        input_pdf_filename = os.path.splitext(os.path.basename(selected_pdf_file))[0]  # 입력 PDF 파일명 추출
        writer = copy_pages(selected_pdf_file, current_batch_copies)

        if writer is None:
            print("PDF 생성에 실패하였습니다.")
            continue
        
        temp_pdf_file = os.path.join("C:\\Users\\LG\\Downloads", "temp.pdf")
        # temp_pdf_file = os.path.join("C:\\Users\\User\\Downloads\\PDF", "temp.pdf")
        with open(temp_pdf_file, "wb") as temp_pdf:
            writer.write(temp_pdf)

        type_choice = select_type()

        # temp.pdf 파일에 워터마크 추가
        writer_with_watermark = add_watermark(temp_pdf_file, type_choice, current_document_number, total_documents)
        new_total_pages = writer_with_watermark.getNumPages()

        # 파일 이름에 입력 PDF 파일명, 타임스탬프, current_document_number 추가
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        output_pdf_file = os.path.join("C:\\Users\\LG\\Downloads\\PDF", f"{input_pdf_filename}_{timestamp}_{current_document_number}.pdf")
        # output_pdf_file = os.path.join("C:\\Users\\User\\Downloads\\PDF", f"{input_pdf_filename}_{timestamp}_{current_document_number}.pdf")
        with open(output_pdf_file, "wb") as output_pdf:
            writer_with_watermark.write(output_pdf)

        print(f"페이지 번호가 추가된 파일이 저장되었습니다: {output_pdf_file}")

        # 다음 문서 번호로 업데이트
        # current_document_number += current_batch_copies + (1 if writer.getNumPages() == 2 else 0)
        current_document_number += new_total_pages

    print("모든 문서 번호가 추가되었습니다.")

if __name__ == "__main__":
    main()
