;; 문서 경로에 24개 dwg 저장됨

; 함수 정의: 다수의 영역 추출 및 각각 파일로 저장
(defun c:ExtractMultipleRegions ()
 (setq coords '(
   ;; 1행 1열 (-10133 158) 여백 35 도면 8
   (-4036 158 -3200 740) (-4907 158 -4071 740) (-5778 158 -4942 740)
   (-6649 158 -5813 740) (-7520 158 -6684 740) (-8391 158 -7555 740)
   (-9262 158 -8426 740) (-10133 158 -9297 740)
   ;; 2행 1열 (-10133 -450) 여백 35 도면 8
   (-4036 -450 -3200 132) (-4907 -450 -4071 132) (-5778 -450 -4942 132)
   (-6649 -450 -5813 132) (-7520 -450 -6684 132) (-8391 -450 -7555 132)
   (-9262 -450 -8426 132) (-10133 -450 -9297 132)
   ;; 3행 1열 (-10133 -1096) 여백 35 도면 8
   (-4036 -1096 -3200 -514) (-4907 -1096 -4071 -514) (-5778 -1096 -4942 -514)
   (-6649 -1096 -5813 -514) (-7520 -1096 -6684 -514) (-8391 -1096 -7555 -514)
   (-9262 -1096 -8426 -514) (-10133 -1096 -9297 -514)
   ;; 추가 좌표 반복...
   ))
​
 (foreach coord coords
   (setq min_x (car coord) min_y (cadr coord) max_x (caddr coord) max_y (cadddr coord))
   (setq ss (ssget "W" (list min_x min_y) (list max_x max_y)))
   (if (/= ss nil)
     (progn
       (command "_-WBLOCK" (strcat "output_" (itoa (1+ (setq index (if index (+ index 1) 0))))) "" "0,0" ss "")
       (princ (strcat "\nSaved region " (itoa (1+ index)) "."))
     )
     (princ "\nNo objects selected in this region.")
   )
 )
 (princ)
)
​
; 명령어 등록
(princ "\nExtractMultipleRegions 명령어를 사용하여 다수 영역을 추출하세요. (명령어: ExtractMultipleRegions)")
(princ)
​​