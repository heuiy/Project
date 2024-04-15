(defun c:ExportLayoutsToPDF ()
 (vl-load-com)
 (setq doc (vla-get-ActiveDocument (vlax-get-acad-object)))
​
 (vlax-for layout (vla-get-Layouts doc)
   (setq name (vla-get-name layout))
   (if (not (wcmatch name "Model")) ; "Model" 레이아웃은 제외합니다.
     (progn
       (vla-put-ActiveLayout doc layout) ; 레이아웃을 활성화합니다.
       (command "-PLOT" "Y") ; "-PLOT" 명령어로 PDF로 인쇄합니다.
       ; 인쇄 설정에 따라 추가적인 명령 옵션을 입력해야 할 수 있습니다.
     )
   )
 )
 (princ)
)
​
; AutoCAD에서 이 함수를 실행하기 위해 "ExportLayoutsToPDF"를 호출합니다.
​​
