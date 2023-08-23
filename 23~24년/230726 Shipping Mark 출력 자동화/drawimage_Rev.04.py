# temp.pdf 로 나눠서 진행
# 일단 연속된 숫자로 워터마크 인쇄됨

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

def copy_pages(input_pdf_path, copies):
    reader = PdfFileReader(input_pdf_path)
    writer = PdfFileWriter()
    for _ in range(copies):
        for i in range(reader.getNumPages()):
            writer.addPage(reader.getPage(i))
    return writer

def add_watermark(input_pdf_path, type):
    positions = {
        'UNICEF': (210, 80),
        'SPA': (298, 44),
        'JPNY': (234, 222),
        'EUHR': (248, 38),
        'CHNFA': (394, 326),
        'CHNF': (248, 44),
        'ENG': (248, 44)
    }
    reader = PdfFileReader(input_pdf_path)
    writer = PdfFileWriter()
    x_position, y_position = positions[type]
    total_pages = reader.getNumPages()
    for i in range(total_pages):
        page = reader.getPage(i)
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=letter)
        page_number_text = f"{i + 1}              {total_pages}"
        can.drawString(x_position, y_position, page_number_text)
        can.save()
        packet.seek(0)
        watermark = PdfFileReader(packet)
        page.mergePage(watermark.getPage(0))
        writer.addPage(page)
    return writer

def main():
    selected_pdf_file = select_pdf_file()
    copies = int(input("복사할 페이지 수를 입력하세요: "))
    type_choice = input("유형을 입력하세요 (UNICEF, SPA, JPNY, EUHR, CHNFA, CHNF, ENG 중 하나): ")

    temp_pdf_file = os.path.join("C:\\Users\\LG\\Downloads", "temp.pdf")

    # 원본 PDF 파일을 복사하여 temp.pdf로 저장
    writer = copy_pages(selected_pdf_file, copies)
    with open(temp_pdf_file, "wb") as temp_pdf:
        writer.write(temp_pdf)

    # temp.pdf 파일에 워터마크 추가
    writer_with_watermark = add_watermark(temp_pdf_file, type_choice)

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    output_pdf_file = os.path.join("C:\\Users\\LG\\Downloads", f"{timestamp}.pdf")
    with open(output_pdf_file, "wb") as output_pdf:
        writer_with_watermark.write(output_pdf)

    print(f"페이지 번호가 추가된 파일이 저장되었습니다: {output_pdf_file}")

if __name__ == "__main__":
    main()
