# 30여장을 pdf 한 페이지에 출력했음
# pdf 한 페이지에 도면 하나씩 나오도록 해야 함
# 색깔도 약간 알록달록한데 흑백으로 나와야 함
# 사용자에 따라 출력 조건이 조정될 수 있어야 함

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
           # 모델 스페이스에서 모든 객체를 그림
           fig = plt.figure()
           ax = fig.add_axes([0, 0, 1, 1])
           ax.axis('off')
           Frontend(ctx, MatplotlibBackend(ax)).draw_layout(msp, finalize=True)
           pdf.savefig(fig)
           plt.close(fig)

