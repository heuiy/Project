(defun c:CreateLayoutsFromCoordinates ()
 (setq doc (vla-get-activedocument (vlax-get-acad-object)))  ; 현재 문서 객체를 가져옵니다.
 (setq modelSpace (vla-get-modelspace doc))                 ; 모델 공간을 가져옵니다.
 (setq layouts (vla-get-layouts doc))                       ; 레이아웃 컬렉션을 가져옵니다.
​
 ; 좌표 목록을 설정합니다.
 (setq drawingsCoords '(
   (-4036 158 -3200 740)
   (-4907 158 -4071 740)
   (-5778 158 -4942 740)
   (-6649 158 -5813 740)
   (-7520 158 -6684 740)
   (-8391 158 -7555 740)
   (-9262 158 -8426 740)
   (-10133 158 -9297 740)
   ; 추가 좌표 입력...
 ))
​
 (foreach coord drawingsCoords
   (setq min_x (car coord) min_y (cadr coord) max_x (caddr coord) max_y (cadddr coord))
   (setq ss (ssget "C" (list min_x min_y) (list max_x max_y)))
​
   ; 새로운 레이아웃을 생성합니다.
   (setq layoutName (strcat "Layout_" (itoa (1+ (setq index (if index (+ index 1) 0))))))
   (setq newLayout (vla-add layouts layoutName))
​
   ; 선택된 객체들을 새로운 레이아웃의 블록으로 복사합니다.
   (vlax-for obj ss
     (setq destinationBlock (vla-get-block newLayout))   ; 새 레이아웃의 블록을 가져옵니다.
     (vla-copyobj obj destinationBlock)                   ; 객체를 새 레이아웃의 블록으로 복사합니다.
   )
​
   ; 배치 활성화
   (vla-put-activatelayout doc newLayout)
​
   (princ (strcat "\nCreated layout for region: " layoutName))
 )
 (princ)
)
​
(princ "\nType 'CreateLayoutsFromCoordinates' to run this function.\n")
​​