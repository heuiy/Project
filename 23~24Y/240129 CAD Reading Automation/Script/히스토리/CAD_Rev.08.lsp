;; 한 장으로 퉁쳐서 나옴

(defun c:ExportLayoutsToPDF (/ doc name)
 (vl-load-com) ; COM 지원을 위해 불러옵니다.
 (setq doc (vla-get-ActiveDocument (vlax-get-acad-object))) ; 현재 문서를 가져옵니다.
​
 ; 모든 레이아웃을 순회합니다.
 (vlax-for layout (vla-get-Layouts doc)
   (setq name (vla-get-name layout))
   (if (not (wcmatch name "Model")) ; "Model" 레이아웃은 제외합니다.
     (progn
       ; PDF로 내보내기를 실행합니다.
       (command "_.EXPORTPDF" "Y" "" (strcat (getvar "DWGPREFIX") name ".pdf"))
     )
   )
 )
 (princ) ; 함수 종료를 알립니다.
)
​
; 이 함수를 AutoCAD에서 실행하기 위해 `c:ExportLayoutsToPDF`를 호출합니다.
​​
