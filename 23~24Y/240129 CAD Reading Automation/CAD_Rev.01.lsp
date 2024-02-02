(defun c:ExportToPDF ()
   ;; (setq dwgPath "D:/#.Secure Work Folder/BIG/Project/23~24Y/240129 CAD Reading Automation/DWG/in/240119 first/PL1-PD-0000~42_작보1 P&ID_자동화검토용_Rev.01.dwg") ; DWG 파일 경로
   (setq dwgPath "D:/#.Secure Work Folder/BIG/Project/23~24Y/240129 CAD Reading Automation/DWG/in/240119 first/PL1-PD-0000~42_1_P&ID_Rev.01.dwg") ; DWG 파일 경로
   (setq pdfPath "D:/#.Secure Work Folder/BIG/Project/23~24Y/240129 CAD Reading Automation/PDF/out/test.pdf") ; 출력할 PDF 파일 경로
​
   ; 파일을 열고, PDF로 출력
   (command "_open" dwgPath)
   (command "_.-export" "PDF" "_P" "Window"
            (list 20 20) (list 50 50) ; 출력할 영역 설정 (필요에 따라 조정)
            pdfPath "_N" "_Y")
​
   (princ "PDF 변환 완료.")
   (princ)
)
​
(c:ExportToPDF)