# pip install PyPDF2 pdf2image pytesseract pandas

import PyPDF2
from pdf2image import convert_from_path
import pytesseract
import pandas as pd

# PDF 파일 경로
# pdf_path = 'path/to/your/pdf/file.pdf'
pdf_path = 'D:/#.Secure Work Folder/BIG/Project/23~24Y/231122 chart information reading/pdf/in/ARE24008/20231117115139-0001.pdf'

# PDF에서 이미지로 변환
images = convert_from_path(pdf_path)

# 이미지에서 텍스트 추출
data = []
for i, image in enumerate(images):
   text = pytesseract.image_to_string(image)
   # 여기서 각 페이지의 텍스트를 처리하고 RT와 %Area 값을 추출합니다.
   # 이 부분은 실제 PDF 내용과 구조에 따라 맞춤형 코드가 필요할 수 있습니다.
   # 예를 들어, 정규 표현식을 사용하여 특정 패턴을 찾을 수 있습니다.

   # 예시 데이터 추가 (실제로는 추출된 데이터를 사용해야 합니다)
   data.append({'page': i+1, 'RT': '1.234', '%Area': '56.78'})

# 추출된 데이터를 DataFrame으로 변환
df = pd.DataFrame(data)

# 엑셀 파일로 저장
df.to_excel('output.xlsx', index=False)

print('완료되었습니다. 생성된 엑셀 파일을 확인하세요.')