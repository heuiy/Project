;; 새로운 코드

(defun C:ExportToPDF ()
 (setq outputPath "D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\240129 CAD Reading Automation\\PDF\\out\\") ; 출력 경로 설정
 (setq coords '((-6520 -4455 -5684 -3873)
                (-7420 -4455 -6584 -3873)
                (-8320 -4455 -7484 -3873)
                (-9220 -4455 -8384 -3873)
                ))
​
 (foreach coord coords
   (setq x1 (car coord)
         y1 (cadr coord)
         x2 (caddr coord)
         y2 (cadddr coord)
         filename (strcat outputPath "drawing-" (itoa (car coord)) "-" (itoa (cadr coord)) ".pdf")
   )
​
   ;; PDF 출력 명령 실행
   (command "_.-EXPORTPDF" "Window" (list x1 y1) (list x2 y2) filename "")
 )
 (princ)
)
​
;; ExportToPDF 함수 실행
;; (C:ExportToPDF)
​
