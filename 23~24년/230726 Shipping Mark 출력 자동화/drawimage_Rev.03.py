# 자꾸 번호가 겹쳐서 출력됨

from PyPDF4 import PdfFileReader, PdfFileWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from datetime import datetime
import io
import os

def select_pdf_file():
    download_folder = "C:\\Users\\LG\\Downloads"
    pdf_files = [f for f in os.listdir(download_folder) if f.endswith('.pdf')]
    print("사용 가능한 PDF 파일:")
    for idx, file in enumerate(pdf_files):
        print(f"{idx + 1}. {file}")
    choice = int(input("선택할 PDF 파일 번호를 입력하세요: ")) - 1
    return os.path.join(download_folder, pdf_files[choice])

def create_watermark(text, x_position, y_position):
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    can.drawString(x_position, y_position, text)
    can.save()
    packet.seek(0)
    return PdfFileReader(packet)

def add_page_number(input_pdf_path, type, copies):
    reader = PdfFileReader(input_pdf_path)
    writer = PdfFileWriter()

    positions = {
        'UNICEF': (210, 80),
        'SPA': (298, 44),
        'JPNY': (234, 222),
        'EUHR': (248, 38),
        'CHNFA': (394, 326),
        'CHNF': (248, 44),
        'ENG': (248, 44)
    }

    if type not in positions.keys():
        raise ValueError("유효하지 않은 유형입니다. UNICEF, SPA, JPNY, EUHR, CHNFA, CHNF, ENG 중 하나를 선택하세요.")

    x_position, y_position = positions[type]
    total_pages = reader.getNumPages()

    page_number = 1
    for _ in range(copies):
        for i in range(total_pages):
            original_page = reader.getPage(i).createBlankPage()  # 원본 페이지를 가져옴
            page_number_text = "{0}              {1}".format(page_number, total_pages * copies)
            watermark = create_watermark(page_number_text, x_position, y_position)  # 워터마크 생성
            original_page.mergePage(watermark.getPage(0))  # 워터마크 병합
            writer.addPage(original_page)
            page_number += 1

    return writer

def main():
    selected_pdf_file = select_pdf_file()
    copies = int(input("복사할 페이지 수를 입력하세요: "))

    type_options = ['UNICEF', 'SPA', 'JPNY', 'EUHR', 'CHNFA', 'CHNF', 'ENG']
    print("유형을 선택하세요:")
    for idx, option in enumerate(type_options):
        print(f"{idx + 1}. {option}")
    type_choice = int(input("유형 번호를 입력하세요: ")) - 1

    if type_choice < 0 or type_choice >= len(type_options):
        raise ValueError("유효하지 않은 유형 번호입니다.")

    type = type_options[type_choice]

    writer = add_page_number(selected_pdf_file, type, copies)  # copies 인자 추가

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    output_pdf_file = os.path.join("C:\\Users\\LG\\Downloads", f"{timestamp}.pdf")
    with open(output_pdf_file, "wb") as output_pdf:
        writer.write(output_pdf)

    print(f"페이지 번호가 추가된 파일이 저장되었습니다: {output_pdf_file}")

if __name__ == "__main__":
    main()

