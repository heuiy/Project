;; 저장 안됨
;; 1, 2 페이지만 출력됨

(defun C:ExportAreaPDF ()
 (setq coords '((-10127.2527 159.6186 -9306.2257 732.111)
                (-9259.8554 159.6186 -8428.386 732.111)
                (-8390.1987 159.6186 -7557.8397 732.111)
                (-7524.3851 159.6186 -6683.6569 732.111)
                (-6651.6569 159.6186 -5814.2557 732.111)))
​
 (setq outputPath "C:\\Users\\LG\\Downloads\\") ; 출력 경로 설정
 (setq i 0)
​
 (foreach area coords
   (setq xMin (car area)
         yMin (cadr area)
         xMax (caddr area)
         yMax (cadddr area))
​
   (command "-PLOT" "Yes" "Model" "" "DWG To PDF.pc3" "ISO A4 (210.00 x 297.00 MM)" "Millimeters" "Landscape" "No" "Window"
            (strcat (rtos xMin 2 2) "," (rtos yMin 2 2))
            (strcat (rtos xMax 2 2) "," (rtos yMax 2 2))
            "Fit" "Center" "Yes" "monochrome.ctb" "Yes" "No" "No" "No" "Yes"
            "\"" (strcat outputPath "Output_" (itoa (1+ (setq i (1+ i)))) ".pdf") "\""
            "Yes")
​
   (command "_.delay" "2000") ; 2초 지연 추가
 )
 (princ)
)
​​