(defun c:ExtractMultipleRegionsAndExportPDF (/ coords coord min_x min_y max_x max_y ss dwgpath pdfpath index)
 (setq coords '(
   (-4036 158 -3200 740)
   (-4907 158 -4071 740)
   (-5778 158 -4942 740)
   (-6649 158 -5813 740)
   (-7520 158 -6684 740)
   (-8391 158 -7555 740)
   (-9262 158 -8426 740)
   (-10133 158 -9297 740)
   ;; 추가 좌표 입력...
 ))
​
 (foreach coord coords
   (setq min_x (car coord)
         min_y (cadr coord)
         max_x (caddr coord)
         max_y (cadddr coord))
   (setq ss (ssget "W" (list min_x min_y) (list max_x max_y)))
   (if ss
     (progn
       ;; DWG 파일로 먼저 저장
       (setq dwgpath (strcat "D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\240129 CAD Reading Automation\\PDF\\out\\dwg_output_"
         (itoa (1+ (setq index (if index (+ index 1) 0))))))
       (command "_-WBLOCK" "" dwgpath "_non" "0,0" ss "")
       
       ;; PDF로 변환
       (command "_-OPEN" dwgpath)
       (setq pdfpath (strcat "D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\240129 CAD Reading Automation\\PDF\\out\\pdf_output_"
         (itoa index) ".pdf"))
       (command "_-EXPORTPDF" "_Y" pdfpath)
       
       (princ (strcat "\nSaved PDF region " (itoa index) "."))
     )
     (princ "\nNo objects selected in this region.")
   )
 )
 (princ)
)
​
;; 함수를 AutoCAD 명령어로 등록
(princ "\nType 'ExtractMultipleRegionsAndExportPDF' to run this function.\n")
​​