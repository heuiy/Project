# 자투리 배치 뒤로 모으기

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

# 자투리 배치 뒤로 모으기

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

def select_pdf_files():
    download_folder = "C:\\Users\\LG\\Downloads"
    pdf_files = [f for f in os.listdir(download_folder) if f.endswith('.pdf')]
    print("사용 가능한 PDF 파일:")
    for idx, file in enumerate(pdf_files):
        print(f"{idx + 1}. {file}")

    selected_indices = input("선택할 PDF 파일 번호들을 입력하세요 (쉼표로 구분): ").split(',')
    selected_files = [os.path.join(download_folder, pdf_files[int(idx) - 1]) for idx in selected_indices]

    return selected_files

def copy_pages(input_pdf_path, copies):
    reader = PdfFileReader(input_pdf_path)
    writer = PdfFileWriter()
    total_pages = reader.getNumPages()

    if total_pages > 2:
        print("지원하지 않는 페이지 수입니다. 최대 2페이지까지의 PDF만 지원됩니다.")
        return None, None

    for _ in range(copies):
        writer.addPage(reader.getPage(0))

    second_page_writer = None
    if total_pages == 2:
        second_page_writer = PdfFileWriter()
        second_page_writer.addPage(reader.getPage(1))

    return writer, second_page_writer

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
    writer_with_watermark = PdfFileWriter()
    x_position, y_position = positions[type]
    total_pages = reader.getNumPages()
    for i in range(total_pages):
        page = reader.getPage(i)
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=letter)
        page_number_text = f"{current_document_number + i}              {total_documents}"
        can.drawString(x_position, y_position, page_number_text)
        can.save()
        packet.seek(0)
        watermark = PdfFileReader(packet)
        page.mergePage(watermark.getPage(0))
        writer_with_watermark.addPage(page)
    return writer_with_watermark

def select_type():
    type_options = ['STANDARD_CHNE', 'STANDARD_CHNF', 'STANDARD_CHNFA', 'STANDARD_CHNY', 'STANDARD_ENG', 'STANDARD_EUHR', 'STANDARD_JPNY','STANDARD_SPA','STANDARD_UNICEF']
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
    remaining_documents = total_documents  # Initialize the total remaining documents
    second_page_pool = PdfFileWriter()  # To store all second pages

    while current_document_number <= total_documents and remaining_documents > 0:
        print(f"남은 문서 수: {remaining_documents}")

        selected_pdf_files = select_pdf_files()

        for selected_pdf_file in selected_pdf_files:
            input_pdf_filename = os.path.splitext(os.path.basename(selected_pdf_file))[0]
            current_batch_copies = int(input(f"{input_pdf_filename}을(를) 몇 장으로 복사할까요? "))

            if current_batch_copies > remaining_documents:
                print("입력한 배치 페이지 수가 남은 문서 수보다 클 수 없습니다. 다시 시도해주세요.")
                continue

            writer, second_page_writer = copy_pages(selected_pdf_file, current_batch_copies)

            if writer is None:
                print("PDF 생성에 실패하였습니다.")
                continue

            temp_pdf_file = f"C:\\Users\\LG\\Downloads\\TEMP\\temp_{current_document_number}.pdf"
            with open(temp_pdf_file, "wb") as temp_pdf:
                writer.write(temp_pdf)

            # If there is a second page, add it to the pool
            if second_page_writer:
                temp_second_pdf_file = f"C:\\Users\\LG\\Downloads\\TEMP\\temp_second_{current_document_number}.pdf"
                with open(temp_second_pdf_file, "wb") as temp_second_pdf:
                    try:
                        second_page_writer.write(temp_second_pdf)
                    except AttributeError as e:
                        print(f"An error occurred while writing the second page PDF: {e}")
                        continue  # Skip the current iteration and proceed to the next PDF

                # 이후에는 temp_second_pdf_file을 이용하여 추가 작업을 수행할 수 있습니다.
                second_page_pool.addPage(PdfFileReader(temp_second_pdf_file).getPage(0))

            type_choice = select_type()
            writer_with_watermark = add_watermark(temp_pdf_file, type_choice, current_document_number, total_documents)

            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            output_pdf_file = os.path.join("C:\\Users\\LG\\Downloads\\OUT", f"{input_pdf_filename}_{timestamp}_{current_document_number}.pdf")

            # Add exception handling here
            try:
                with open(output_pdf_file, "wb") as output_pdf:
                    writer_with_watermark.write(output_pdf)
            except AttributeError as e:
                print(f"An error occurred while writing the PDF: {e}")
                continue  # Skip the current iteration and proceed to the next PDF

            print(f"페이지 번호가 추가된 파일이 저장되었습니다: {output_pdf_file}")

            new_total_pages = writer.getNumPages()
            current_document_number += new_total_pages
            remaining_documents -= current_batch_copies

    if second_page_pool.getNumPages() > 0:
        output_pdf_file_second_page = os.path.join("C:\\Users\\LG\\Downloads\\OUT", f"all_second_pages_{timestamp}.pdf")
        with open(output_pdf_file_second_page, "wb") as output_pdf:
            second_page_pool.write(output_pdf)

    print("모든 문서 번호가 추가되었습니다.")

if __name__ == "__main__":
    main()
