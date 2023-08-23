# 1장짜리를 50장 이상으로 추가하기

# !pip install pdfrw reportlab
# !pip install PyPDF4

# 몇 개의 구간으로 나누고 싶은지 물어보고,
# 사용자가 전체 페이지수, 시작/끝 페이지수 입력하면 됨
# 사용자가 UNICEF, SPA, JPNY, EUHR, CHNFA, CHNF, ENG 도 선택하면 됨

from PyPDF4 import PdfFileReader, PdfFileWriter, PdfFileMerger
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import os

def select_pdf_file():
    # download_folder = "C:\\Users\\LGChem\\Downloads"
    download_folder = "C:\\Users\\LG\\Downloads"
    pdf_files = [f for f in os.listdir(download_folder) if f.endswith('.pdf')]
    print("사용 가능한 PDF 파일:")
    for idx, file in enumerate(pdf_files):
        print(f"{idx + 1}. {file}")
    choice = int(input("선택할 PDF 파일 번호를 입력하세요: ")) - 1
    return os.path.join(download_folder, pdf_files[choice])

def add_page_number(input_pdf_path, output_pdf_path):
    positions = {
        'UNICEF': (210, 80),
        'SPA': (298, 44),
        'JPNY': (234, 222),
        'EUHR': (248, 38),
        'CHNFA': (394, 326),
        'CHNF': (248, 44),
        'ENG': (248, 44)
    }

    type = input("유형을 선택하세요. UNICEF, SPA, JPNY, EUHR, CHNFA, CHNF, ENG 중 하나를 선택하세요: ")

    if type not in positions.keys():
        raise ValueError("유효하지 않은 유형입니다. UNICEF, SPA, JPNY, EUHR, CHNFA, CHNF, ENG 중 하나를 선택하세요.")

    x_position, y_position = positions[type]

    num_sections = int(input("몇 개의 구간으로 나누시겠습니까?: "))  # 사용자로부터 구간 수를 입력 받음
    temp_files = []  # to hold temporary file names

    for i in range(num_sections):
        reader = PdfFileReader(input_pdf_path)
        writer = PdfFileWriter()

        total_pages = int(input(f"{i+1}. 전체 페이지 수를 입력하세요: "))
        from_page = int(input(f"{i+1}. 시작 페이지 번호를 입력하세요: "))
        to_page = int(input(f"{i+1}. 끝 페이지 번호를 입력하세요: "))

        if from_page < 1 or to_page > total_pages or from_page > to_page:
            raise ValueError("유효하지 않은 페이지 범위입니다.")

        for i in range(from_page, to_page + 1):
            packet = io.BytesIO()

            # We create a new PDF with Reportlab
            can = canvas.Canvas(packet, pagesize=letter)
            page_number_text = "{0}              {1}".format(i, total_pages)
            can.drawString(x_position, y_position, page_number_text)
            can.save()

            # Move to the beginning of the StringIO buffer
            packet.seek(0)

            # Add the "watermark" (which is the new pdf) on the existing page
            page = reader.getPage(i - 1)
            watermark = PdfFileReader(packet)
            page.mergePage(watermark.getPage(0))

            writer.addPage(page)

        temp_file = f"temp{i+1}.pdf"
        temp_files.append(temp_file)
        with open(temp_file, "wb") as output_pdf_file:
            writer.write(output_pdf_file)

    # Merge all the temporary files into the final output file
    merger = PdfFileMerger()
    for temp_file in temp_files:
        merger.append(temp_file)

    merger.write(output_pdf_path)
    merger.close()

    # Delete temporary files
    for temp_file in temp_files:
        os.remove(temp_file)

# 사용자에게 PDF 파일을 선택하게 하고 해당 파일에 페이지 번호 추가
selected_pdf_file = select_pdf_file()
output_pdf_file = os.path.join("C:\\Users\\LGChem\\Downloads", "numbered.pdf")
add_page_number(selected_pdf_file, output_pdf_file)
print(f"페이지 번호가 추가된 파일이 저장되었습니다: {output_pdf_file}")