;; 240416 다시 시작

;; 1장 저장됨, 가로로 출력됨

​
(defun C:EXPORTAREAPDF ()
 ;; 출력 경로 설정
 (setq outputPath "D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\240129 CAD Reading Automation\\PDF\\out\\")
​
 ;; 좌표 목록
 (setq coords '(
   (-4036 158 -3200 740)
   (-4907 158 -4071 740)
   (-5778 158 -4942 740)
   ;; 추가 좌표를 여기에 입력하세요.
 ))
​
 ;; 각 좌표마다 PDF 생성
 (foreach area coords
   (setq ll (list (car area) (cadr area))  ; 하단 왼쪽 좌표
         ur (list (caddr area) (cadddr area))  ; 상단 오른쪽 좌표
   )
   ;; PLOT 명령으로 PDF 출력
   (command "-PLOT"  "Yes"  "" "DWG To PDF.pc3"
            "ISO_A3_(420.00_x_297.00_MM)" "m" "Landscape" "No" "Window"
            ll ur "Fit" "Center" "Yes" "monochrome.ctb" "Yes" "No" "No" "Yes" "No" "No" "Yes"
            (strcat outputPath "Output_" (itoa (1+ (setq i (1+ i)))) ".pdf")
   )
 )
 (princ)
)
​​
​​