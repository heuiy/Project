;; AutoCAD의 명령창에서 PDF라고 입력하고 엔터 키를 누르면, 
;; 스크립트가 시작 페이지 번호와 끝 페이지 번호를 차례로 물어볼 것입니다. 
;; 사용자는 각각의 질문에 대해 숫자를 입력하고 엔터 키를 눌러야 합니다.​

(defun c:PDF ()
 (vl-load-com) ; Visual LISP 함수를 로드합니다.
 (setq doc (vla-get-activedocument (vlax-get-acad-object))) ; 현재 문서를 가져옵니다.
 (setq layouts (vla-get-layouts doc)) ; 모든 레이아웃을 가져옵니다.
​
 ;; 사용자로부터 시작 페이지와 끝 페이지를 입력받습니다.
 (setq start (getint "\n Starting page: "))
 (setq end (getint "\n Ending page: "))
​
 (setq count 1) ; 페이지 카운트를 초기화합니다.
​
 (vlax-for layout layouts
   (if (and (not (wcmatch (vla-get-name layout) "Model")) ; "Model" 레이아웃을 제외합니다.
            (>= count start) ; 시작 페이지 번호보다 크거나 같습니다.
            (<= count end)) ; 끝 페이지 번호보다 작거나 같습니다.
     (progn
       (setq layout-name (vla-get-name layout)) ; 레이아웃 이름을 가져옵니다.
       (setq pdf-name (strcat "C:\\Users\\LG\\Downloads\\" layout-name "out.pdf")) ; PDF 파일명을 설정합니다.
​
       ;; 레이아웃을 현재 레이아웃으로 설정합니다.
       (vla-put-activelayout doc layout)
​
       ;; 플롯 설정을 지정합니다.
       (vla-put-configname layout "DWG To PDF.pc3") ; PDF 프린터 설정을 선택합니다.
       (vla-put-paperunits layout acMillimeters) ; 용지 단위를 설정합니다.
       (vla-put-standardScale layout acScaleToFit) ; 축척을 설정합니다.
       (vla-put-centerplot layout :vlax-true) ; 중앙에 플롯팅합니다.
       (vla-put-plotrotation layout ac0degrees) ; 플롯 회전을 설정합니다.
       (vla-put-plottype layout acExtents) ; 플롯 유형을 설정합니다.
       (vla-put-usestandardScale layout :vlax-true) ; 표준 축척 사용을 설정합니다.
       (vla-put-canonicalmedianame layout "ISO_full_bleed_A3_(420.00_x_297.00_MM)") ; 용지 크기를 설정합니다.
​
       ;; PDF로 내보내기 위한 명령어 실행
       (command "_.-export" "_PDF" "Y" "" pdf-name) ; PDF로 내보냅니다.
     )
   )
   (setq count (+ count 1)) ; 페이지 카운트를 증가시킵니다.
 )
 (princ (strcat "\nPage " (itoa start) " to " (itoa end) " are transformed to PDF."))
 (princ)
)
​​
