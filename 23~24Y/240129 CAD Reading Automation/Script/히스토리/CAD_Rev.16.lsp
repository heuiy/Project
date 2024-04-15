;; 

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
   (command "-PLOT" "Y" "Model" "" "DWG To PDF.pc3" "" "Inches" "Landscape" "N" "Window"
            (strcat (rtos xMin 2 2) "," (rtos yMin 2 2))
            (strcat (rtos xMax 2 2) "," (rtos yMax 2 2))
            "Fit" "Center" "Y" "monochrome.ctb" "Y" "N" "N" "N" "Y"
            (strcat outputPath "Output_" (itoa (1+ (setq i (1+ i)))) ".pdf")
            "Y")
​
   (command "_.delay" "5000") ; 5초 지연 추가
 )
 (princ)
)
​​