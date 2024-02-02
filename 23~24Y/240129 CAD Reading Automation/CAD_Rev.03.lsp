(defun c:ExportAllLayoutsToPDF ()
 (vl-load-com) ; Visual LISP 함수를 로드합니다.
 (setq layouts (vla-get-layouts (vla-get-activedocument (vlax-get-acad-object)))) ; 모든 레이아웃을 가져옵니다.
 (vlax-for layout layouts
   (unless (wcmatch (vla-get-name layout) "Model")
     (setq layout-name (vla-get-name layout)) ; 레이아웃 이름을 가져옵니다.
    ;;  (setq pdf-name (strcat "D:/PDFs/" layout-name ".pdf")) ; PDF 파일명을 설정합니다. 저장 경로는 적절히 수정해주세요.
    ;;  (setq pdfPath "D:/#.Secure Work Folder/BIG/Project/23~24Y/240129 CAD Reading Automation/PDF/out/test.pdf") ; 출력할 PDF 파일 경로     
    ;;  (setq pdf-name (strcat "D:/#.Secure Work Folder/BIG/Project/23~24Y/240129 CAD Reading Automation/PDF/out/" layout-name "test.pdf")) ; PDF 파일명을 설정합니다. 저장 경로는 적절히 수정해주세요.
     (setq pdf-name (strcat "C:/Users/LG/Downloads/" layout-name "test.pdf")) ; PDF 파일명을 설정합니다. 저장 경로는 적절히 수정해주세요.     
     (vla-put-configname layout "DWG To PDF.pc3") ; PDF 프린터 설정을 선택합니다.
     (vla-put-paperunits layout acMillimeters) ; 용지 단위를 설정합니다.
     (vla-put-standardScale layout acScaleToFit) ; 축척을 설정합니다.
     (vla-put-centerplot layout :vlax-true) ; 중앙에 플롯팅합니다.
     (vla-put-plotrotation layout ac0degrees) ; 플롯 회전을 설정합니다.
     (vla-put-plottype layout acExtents) ; 플롯 유형을 설정합니다.
     (vla-put-usestandardScale layout :vlax-true) ; 표준 축척 사용을 설정합니다.
     (vla-put-canonicalmedianame layout "ISO_full_bleed_A3_(420.00_x_297.00_MM)") ; 용지 크기를 설정합니다.
     (vla-plot layout) ; 플롯팅합니다.
     (vla-exportlayout layout pdf-name) ; PDF로 내보냅니다.
   )
 )
 (princ "All layouts are transformed to PDF")
 (princ)
)