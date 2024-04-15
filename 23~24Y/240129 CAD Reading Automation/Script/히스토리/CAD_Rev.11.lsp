;; 3D CAD Pro 에 질의함

(defun C:ExportAreaPDF ()
 (setq coords '((-4058.7059 143.0281 -3183.1443 751.1082)
                (-4926.8476 128.1969 -4051.286 743.6926)
                (-5794.9893 135.6125 -4949.1077 743.6926)))
;;  (setq coords '((-3183.1443 751.1082 -4058.7059 143.0281)
;;                 (-4051.286 743.6926 -4926.8476 128.1969)
;;                 (-4949.1077 743.6926 -5794.9893 135.6125)))
 (setq outputPath "D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\240129 CAD Reading Automation\\PDF\\in\\") ; 출력 경로 설정
​
 (setq i 0)
 (foreach area coords
   (setq xMin (car area)
         yMin (cadr area)
         xMax (caddr area)
         yMax (cadddr area))
​
   (command "_-PLOT" "Y" ; Detailed plot configuration
                    "Model" ; Plot area
                    "DWG To PDF.pc3" ; Plotter name
                    "ISO_full_bleed_A3_(420.00_x_297.00_MM)" ; Paper size
                    "M" ; Plot scale
                    "W" ; Window
                    (strcat (rtos xMin 2 2) "," (rtos yMin 2 2)) ; Window lower left corner
                    (strcat (rtos xMax 2 2) "," (rtos yMax 2 2)) ; Window upper right corner
                    "N" ; Plot style table
                    "Y" ; Lineweights
                    "N" ; PDF options
                    "N" ; Save changes to layout
                    "Y" ; Proceed with plot
                    (strcat outputPath "Output_" (itoa (1+ (setq i (1+ i)))) ".pdf") ; Output file name
   )
 )
 (princ)
)
​​