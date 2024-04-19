(defun C:ExportAreaPDF ()
 ;; 출력 경로 설정
 (setq outputPath "C:\\Users\\LG\\Desktop\\")
 
 ;; 도면 좌표 설정
 (setq coords '(
   (-5620 -5720 -4784 -5138)
   (-6520 -5720 -5684 -5138)
   (-7420 -5720 -6584 -5138)
   (-8320 -5720 -7484 -5138)
   (-9220 -5720 -8384 -5138)
 ))

 ;; 각 좌표에 대해 PDF 출력
 (foreach area coords
   (setq ll (list (car area) (cadr area)))  ; 하단 왼쪽 좌표
   (setq ur (list (caddr area) (cadddr area)))  ; 상단 오른쪽 좌표

   ;; 플롯 명령 실행
   (command "-PLOT" "Y" "" "DWG To PDF.pc3" "ISO_A3_(420.00_x_297.00_MM)" "m" "Landscape" "No" "Window"
            ll ur "Fit" "Center" "Yes" "monochrome.ctb" "Yes" "No" "No" "Yes" "No" "No" "Yes"
            (strcat outputPath "Output_" (itoa (1+ (setq i 1))) ".pdf")
   )
   (command "_.delay" "5000")  ; 대기
 )

 (princ "\nPDF export completed.")
 (princ)
)

;; 코드를 로드하고 AutoCAD에서 C:ExportAreaPDF 명령어를 사용하여 실행하세요.
