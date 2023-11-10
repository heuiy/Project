# 자투리 배치 뒤로 모으기 (임시 빽업!!)

# 자투리 파일 삽입 순서는 제대로!!
# 자투리 파일 페이지 번호는 역순!!
# 자투리 파일 모은 것은 끝에서부터 번호가 지정될 수밖에 없음
# 2 페이지인 pdf 가 몇 개 입력될지 모르기 때문에
# 시작 페이지를 알 수 없음

# 끝 번호를 지정해서 페이지 추가될 때마다 1장씩 번호를 빼야 함

# 자투리 파일 페이지 번호도 순서대로 표기하려면
# 단계를 나눠야 함

'''
- 2번째 페이지 모은 pdf 는 워터마크 번호 추가하지 말고 저장만 하기
- 해당 페이지 불러와서 사용자가 시작 페이지와 끝 페이지 지정해주면 
- 오름차순으로 해당 페이지에 워터 마크 추가하도록
'''

# pdf 유형은 한 번만 물어보도록 수정
# 자투리 모은 pdf 가 순서대로 추가되도록 수정
    # 기존 코드에서는 먼저 second_page_pool에 있는 모든 페이지를 previous_pages로 저장하고,
    # 새로운 두 번째 페이지를 추가한 후
    # 다시 previous_pages를 second_page_pool에 추가하고 있음.
    # 이로 인해 순서가 역순이 되었음

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

'''
메인 작업 흐름
│
├── 변수 초기화 (total_documents, current_document_number 등)
│
└── remaining_documents <= 0이 될 때까지 반복
    │
    ├── 다운로드 폴더에서 사용 가능한 PDF 파일 표시
    │
    ├── 사용자가 PDF 파일 선택
    │
    ├── 선택된 PDF 파일을 순회하는 반복문
    │   │
    │   ├── 사용자에게 현재 PDF의 전체 페이지 수를 묻기
    │   │
    │   ├── 필요한 복사본의 수를 사용자에게 묻기
    │   │
    │   ├── 첫 번째 페이지 (그리고 가능하다면 두 번째 페이지)의 복사본 생성
    │   │
    │   ├── 복사본을 임시로 저장하기
    │   │
    │   ├── 워터마크 유형을 선택하라고 사용자에게 묻기
    │   │
    │   ├── 복사된 페이지에 워터마크 (페이지 번호) 추가
    │   │
    │   ├── 워터마크가 있는 새 PDF 저장
    │   │
    │   └── remaining_documents, current_document_number 업데이트
    │
    └── 두 번째 페이지가 있을 경우
        │
        ├── 모든 두 번째 페이지를 결합
        │
        └── 결합된 두 번째 페이지를 새 PDF로 저장

- select_pdf_files: 사용자가 다운로드 폴더에서 사용할 PDF 파일을 선택하게 합니다.
- copy_pages: 선택된 PDF 파일의 첫 번째 페이지 (및 두 번째 페이지)를 복사하여 새 PDF Writer 객체를 생성합니다.
- add_watermark: 선택된 타입에 따라 각 페이지에 워터마크 (페이지 번호)를 추가합니다.
- select_type: 워터마크 유형을 선택합니다.
'''

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
    current_document_number = 1  # 첫 번째 페이지에 대한 현재 문서 번호
    second_page_document_number = total_documents  # 두 번째 페이지에 대한 현재 문서 번호
    remaining_documents = total_documents

    second_page_pool = PdfWriter()

    while remaining_documents > 0:
        print(f"남은 문서 수: {remaining_documents}")
        selected_pdf_files = select_pdf_files()

        for selected_pdf_file in selected_pdf_files:
            input_pdf_filename = os.path.splitext(os.path.basename(selected_pdf_file))[0]
            total_pages = int(input(f"{input_pdf_filename}의 전체 페이지 수는 몇 장인가요? (1 또는 2): "))
            first_page_copies = int(input(f"{input_pdf_filename}를 몇 장 복사할까요? "))

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

            if total_pages == 2:
                second_page_temp_pdf_file = f"C:\\Users\\LG\\Downloads\\TEMP\\temp_second_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pdf"
                with open(second_page_temp_pdf_file, "wb") as temp_second_pdf:
                    second_page_writer.write(temp_second_pdf)

                second_page_with_watermark = add_watermark(second_page_temp_pdf_file, type_choice,
                                                           second_page_document_number, total_documents)

                # 이 PDF 파일에 대한 두 번째 페이지를 second_page_pool에 추가
                second_page_pool.add_page(second_page_with_watermark.pages[0])

                # 두 번째 페이지에 대한 문서 번호 감소
                second_page_document_number -= 1

            # 첫 번째 페이지에 대한 문서 번호 증가 및 남은 문서 수 감소
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
