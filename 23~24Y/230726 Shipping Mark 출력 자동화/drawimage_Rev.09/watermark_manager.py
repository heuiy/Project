from PyPDF4 import PdfFileReader, PdfFileWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

def add_watermark(input_pdf_path, type, current_document_number, total_documents):
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
        page_number_text = f"{current_document_number + i}              {total_documents}"
        can.drawString(x_position, y_position, page_number_text)
        can.save()
        packet.seek(0)
        watermark = PdfFileReader(packet)
        page.mergePage(watermark.getPage(0))
        writer.addPage(page)
    return writer
