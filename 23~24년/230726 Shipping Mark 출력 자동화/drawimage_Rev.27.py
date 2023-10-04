# 자투리 배치 뒤로 모으기

# 일단 페이지 수까지 정상 출력됨

# TEMP 폴더 하나 만들기
# 그 안에 임시 파일이 생성됨

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

from PyPDF2 import PdfReader, PdfWriter
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import io

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
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()
    total_pages = len(reader.pages)

    if total_pages > 2:
        print("지원하지 않는 페이지 수입니다. 최대 2페이지까지의 PDF만 지원됩니다.")
        return None, None

    for _ in range(copies):
        writer.add_page(reader.pages[0])

    second_page_writer = None
    if total_pages == 2:
        second_page_writer = PdfWriter()
        second_page_writer.add_page(reader.pages[1])

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
    reader = PdfReader(input_pdf_path)
    writer_with_watermark = PdfWriter()
    x_position, y_position = positions[type]
    total_pages = len(reader.pages)
    for i in range(total_pages):
        page = reader.pages[i]
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=letter)
        page_number_text = f"{current_document_number + i}              {total_documents}"
        can.drawString(x_position, y_position, page_number_text)
        can.save()
        packet.seek(0)
        watermark = PdfReader(packet)
        page.merge_page(watermark.pages[0])
        writer_with_watermark.add_page(page)
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
    remaining_documents = total_documents
    second_page_pool = PdfWriter()
    second_page_start_number = total_documents  # second_page_pool의 시작 번호 초기화

    while remaining_documents > 0:
        print(f"남은 문서 수: {remaining_documents}")

        selected_pdf_files = select_pdf_files()

        for selected_pdf_file in selected_pdf_files:
            input_pdf_filename = os.path.splitext(os.path.basename(selected_pdf_file))[0]
            total_pages = int(input(f"{input_pdf_filename}의 전체 페이지 수는 몇 장인가요? (1 또는 2): "))

            first_page_copies = int(input(f"{input_pdf_filename}의 첫 페이지를 몇 장 복사할까요? "))

            if total_pages == 2:
                first_page_copies -= 1
                remaining_documents -= 1  # 두 번째 페이지를 위해 전체 문서 수에서 1을 빼줌

            if first_page_copies > remaining_documents:
                print("복사할 페이지 수가 남은 문서 수보다 클 수 없습니다.")
                continue

            writer, second_page_writer = copy_pages(selected_pdf_file, first_page_copies)

            if writer is None:
                print("PDF 생성에 실패하였습니다.")
                continue

            temp_pdf_file = f"C:\\Users\\LG\\Downloads\\TEMP\\temp_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pdf"
            with open(temp_pdf_file, "wb") as temp_pdf:
                writer.write(temp_pdf)

            type_choice = select_type()
            writer_with_watermark = add_watermark(temp_pdf_file, type_choice, current_document_number, total_documents)

            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            output_pdf_file = os.path.join("C:\\Users\\LG\\Downloads\\OUT",
                                           f"{input_pdf_filename}_{timestamp}_{current_document_number}.pdf")

            with open(output_pdf_file, "wb") as output_pdf:
                writer_with_watermark.write(output_pdf)

            # 두 번째 페이지가 있다면
            if total_pages == 2:
                second_page_temp_pdf_file = f"C:\\Users\\LG\\Downloads\\TEMP\\temp_second_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pdf"
                with open(second_page_temp_pdf_file, "wb") as temp_second_pdf:
                    second_page_writer.write(temp_second_pdf)

                # 수정된 부분: 두 번째 페이지의 워터마크 페이지 번호를 설정
                # 두 번째 페이지가 추가될 때마다 순서를 반대로 하여 올바른 순서를 얻습니다.
                second_page_number = total_documents - len(second_page_pool.pages)

                type_choice_second_page = select_type()
                second_page_with_watermark = add_watermark(second_page_temp_pdf_file, type_choice_second_page,
                                                           second_page_number, total_documents)
                # 페이지를 추가하기 전에 이전 페이지를 저장
                previous_pages = [page for page in second_page_pool.pages]

                # 새 페이지를 추가
                second_page_pool = PdfWriter()
                second_page_pool.add_page(second_page_with_watermark.pages[0])

                # 이전 페이지를 다시 추가 (이렇게 하면 새 페이지가 마지막에 추가됩니다)
                for page in previous_pages:
                    second_page_pool.add_page(page)

            current_document_number += first_page_copies
            remaining_documents -= first_page_copies

            print(f"페이지 번호가 추가된 파일이 저장되었습니다: {output_pdf_file}")

            if remaining_documents <= 0:
                break

        if remaining_documents <= 0:
            break

    if len(second_page_pool.pages) > 0:
        output_pdf_file_second_page = os.path.join("C:\\Users\\LG\\Downloads\\OUT", f"all_second_pages_{timestamp}.pdf")
        with open(output_pdf_file_second_page, "wb") as output_pdf:
            second_page_pool.write(output_pdf)

    print("모든 문서 번호가 추가되었습니다.")

if __name__ == "__main__":
    main()
