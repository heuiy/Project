# 여전히 pdf 1장에 수 많은 도면이 한 번에 출력됨

import os
import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from matplotlib.backends.backend_pdf import PdfPages

# 변환된 DXF 파일이 있는 폴더 경로
dxf_folder_path = r"D:\#.Secure Work Folder\BIG\Project\23~24Y\240129 CAD Reading Automation\DXF\in\240119 first"

# PDF를 저장할 폴더 경로
pdf_folder_path = r"D:\#.Secure Work Folder\BIG\Project\23~24Y\240129 CAD Reading Automation\PDF\in\240119 first"

# 폴더 내의 모든 DXF 파일을 순회
for filename in os.listdir(dxf_folder_path):
   if filename.lower().endswith(".dxf"):
       dxf_file_path = os.path.join(dxf_folder_path, filename)
       pdf_file_name = os.path.splitext(filename)[0] + ".pdf"
       pdf_file_path = os.path.join(pdf_folder_path, pdf_file_name)

       # DXF 파일 읽기
       doc = ezdxf.readfile(dxf_file_path)
       msp = doc.modelspace()

       # 렌더링 컨텍스트 생성
       ctx = RenderContext(doc)

       # Matplotlib를 사용하여 PDF 페이지 생성
       with PdfPages(pdf_file_path) as pdf:
           for layout in doc.layouts:
               # 'Model' 레이아웃을 포함하여 모든 레이아웃 처리
               paperspace = doc.layout(layout.name)

               # 페이지 설정
               fig = plt.figure()
               ax = fig.add_axes([0, 0, 1, 1])
               ax.axis('off')

               # 렌더링을 위한 백엔드 설정
               backend = MatplotlibBackend(ax)

               # layout의 모든 객체를 그림
               Frontend(ctx, backend).draw_layout(paperspace, finalize=True)

               # PDF 페이지에 저장
               pdf.savefig(fig, bbox_inches='tight')
               plt.close(fig)