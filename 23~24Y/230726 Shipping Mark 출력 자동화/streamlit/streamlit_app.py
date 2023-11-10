import streamlit as st
from PyPDF4 import PdfFileReader, PdfFileWriter, PdfFileMerger
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import os

def add_page_number(input_pdf_path, output_pdf_path, type, sections):
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
    
    temp_files = []  # to hold temporary file names

    for i, section in enumerate(sections):
        reader = PdfFileReader(input_pdf_path)
        writer = PdfFileWriter()

        total_pages, from_page, to_page = section

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

# Streamlit 코드
st.title('PDF 페이지 번호 추가')

input_pdf = st.file_uploader("PDF 파일 선택", type=['pdf'])
output_pdf_path = st.text_input("결과 PDF 파일 경로를 입력하세요:")
type = st.selectbox("유형을 선택하세요.", ['UNICEF', 'SPA', 'JPNY', 'EUHR', 'CHNFA', 'CHNF', 'ENG'])

num_sections = st.number_input("구간 수를 입력하세요:", min_value=1, step=1)
sections = []
for i in range(num_sections):
    total_pages = st.number_input(f"{i+1}. 전체 페이지 수를 입력하세요:", min_value=1, step=1)
    from_page = st.number_input(f"{i+1}. 시작 페이지 번호를 입력하세요:", min_value=1, step=1)
    to_page = st.number_input(f"{i+1}. 끝 페이지 번호를 입력하세요:", min_value=1, step=1)
    sections.append((total_pages, from_page, to_page))

if st.button("PDF 페이지 번호 추가"):
    if input_pdf is None or output_pdf_path == "":
        st.warning("모든 필드를 채워주세요.")
    else:
        with open("temp_input.pdf", "wb") as f:
            f.write(input_pdf.read())
        add_page_number("temp_input.pdf", output_pdf_path, type, sections)
        os.remove("temp_input.pdf")
        st.success("페이지 번호 추가 완료!")                                