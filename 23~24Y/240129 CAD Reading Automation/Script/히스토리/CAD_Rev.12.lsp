;; 작은 성공, 특정 도면 pdf 출력 가능함!!!

(defun C:ExportAreaPDF ()
 (setq coords '((-4058.7059 143.0281 -3183.1443 751.1082)
                (-4926.8476 128.1969 -4051.286 743.6926)
                (-5794.9893 135.6125 -4949.1077 743.6926)))
​
 (setq outputPath "D:\\Secure_Work_Folder\\BIG\\Project\\23_24Y\\240129_CAD_Reading_Automation\\PDF\\in\\") ; 출력 경로 설정
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
            "Fit" "Center" "Y" "" "Y" "N" "N" "N" "Y"
            (strcat outputPath "Output_" (itoa (1+ (setq i (1+ i)))) ".pdf")
            "Y")
 )
 (princ)
)
​​