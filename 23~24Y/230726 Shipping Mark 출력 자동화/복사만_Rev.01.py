from PyPDF4 import PdfFileReader, PdfFileWriter
import os

def select_pdf_file():
    download_folder = "C:\\Users\\LG\\Downloads"  # 다운로드 폴더 경로
    pdf_files = [f for f in os.listdir(download_folder) if f.endswith('.pdf')]
    print("사용 가능한 PDF 파일:")
    for idx, file in enumerate(pdf_files):
        print(f"{idx + 1}. {file}")
    choice = int(input("선택할 PDF 파일 번호를 입력하세요: ")) - 1
    return os.path.join(download_folder, pdf_files[choice])

def copy_pages(input_pdf_path, copies):
    reader = PdfFileReader(input_pdf_path)
    writer = PdfFileWriter()

    total_pages = reader.getNumPages()

    for _ in range(copies):
        for i in range(total_pages):
            page = reader.getPage(i)
            writer.addPage(page)

    return writer

def main():
    selected_pdf_file = select_pdf_file()
    copies = int(input("복사할 페이지 수를 입력하세요: "))

    writer = copy_pages(selected_pdf_file, copies)

    output_pdf_file = os.path.join("C:\\Users\\LG\\Downloads", "temp.pdf")  # 파일명을 'temp.pdf'로 변경
    with open(output_pdf_file, "wb") as output_pdf:
        writer.write(output_pdf)

    print(f"파일이 저장되었습니다: {output_pdf_file}")

if __name__ == "__main__":
    main()
