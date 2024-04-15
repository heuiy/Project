;;==================================================================
;;    도면 내 여러 블럭 선택하여 스케일 맞게 자동 플롯 하기
;;    BLOG http://kimhyunchul.co.kr , blog.naver.com/iagapeu
;;    2017. 01. 05. A3용지 스케일에 맞춰 출력하도록 업데이트 
;;    2018. 07. 24. 가로/세로에 따라 용지 자동 설정하게 변경
;;    2018. 11. 02. 프린터 플롯 및 PDF 출력 통합
;;    2018. 11. 08. 버그 수정
;;    2018. 11. 13. 모형탭 출력 추가 용지출력 mmm, pdf저장 pdfm
;;    2019. 07. 29. osnap 풀리는 현상 수정함 (osmode 4335)
;;    2019. 11. 18. 배치탭에서 선가중치(CTB) 스케일로 적용 되는 부분 수정 (선가중치플롯 축척에 적용을 Y 에서 N으로)
;;    2019. 11. 19. A4용지 출력 되도록 (추가명령 ppp4, pdf4, mmmm4, pdfm4)
;;    2020. 03. 26. 배치탭 글자 외곽만 보이는 현상 수정 (도면공간객체숨기기 N으로 변경)
;;    2020. 12. 28. UCS 바뀌어도 출려 가능하도록 수정 (별명없음님 팁) 
;;    2022. 02. 09. 옵션 추가 스케일 용지에맞춤(f) 옵션, 모형탭 도면공간 먼저 플롯 선택여부 및 도면 공간 객체숨기기 옵션 선택, PLOTTRANSPARENCYOVERRIDE 값 옵션
;;    2022. 03. 27. 배치탭 A4 출력 FIT 설정 부분 오타 수정 
;;    2023. 12. 28. zw CAD, Gstar CAD 호환 추가 / A4 인쇄 세팅 수정 / 출력하기 메뉴 (명령어 pppp) 추가
;;    2024. 02. 01. Adobe PDF 프린터 가능하도록 수정 (filedia 조정)
;;    2024. 02. 07. A4 용지 FIT 오류 수정
;;==================================================================

;;본 리습 파일의 저장 위치 기본값은 c:\PDF\PPP 폴더 입니다.

;;아래는 종이로 출력하는 용도의 세팅입니다.
 (setq plotername "HP 000 Printer"); 출력할 프린터이름
 (setq ctbname "acad.ctb") ; A3 출력할 CTB 종류
 (setq ctbnameA4 "acad.ctb") ; A4 출력할 CTB 종류
 (setq papername "A3") ; 용지 크기는 A3 종이 종류 선택, 프린터마다 상태 다르니 확인바람. ex A3 , A3(297x420mm)
 (setq papernamea4 "A4") ; 용지 크기는 A4 종이 종류 선택, 프린터마다 상태 다르니 확인바람. ex A4, A4(210x297mm)
			 ;예) ISO 전체페이지 A3(420.00 x 297.00 mm) 등으로 설정됨
 (setq pdgsccpaper 514.4016); 용지 대각선 길이 A3기준임(도면 폼 크기 틀릴 경우 여기를 고쳐주면 됨)
 (setq pdgsccpapera4 363.7430); 용지 대각선 길이 A4기준임(도면 폼 크기 틀릴 경우 여기를 고쳐주면 됨)


 ;;아래는 PDF로 출력하는 용도의 세팅입니다. CTB는 종이 출력 용도와 동일하게 따라갑니다.
 (setq P_plotername "DWG to PDF.pc5");PDF출력할 플롯터 이름 (Adobe PDF.pc3은 파일 저장 문제로 잘 안됩니다.)
 (setq P_papername "ISO expand A3 (420.00 x 297.00 MM)") ; 용지 크기는 A3 종이 종류 선택, 프린터마다 상태 다르니 확인바람. 
 (setq P_papernamea4 "ISO expand A4 (297.00 x 210.00 MM)") ; 용지 크기는 A3 종이 종류 선택, 프린터마다 상태 다르니 확인바람. 
			 ;예) ISO 전체페이지 A3(420.00 x 297.00 mm) 등으로 설정됨
 (setq PATH (getvar "c:\\pdf\\"))	;;; pdf파일 저장 위치 현재는 c:\pdf 폴더임. 폴더 없을 경우는 만드세요.
 (setq PDFdelay 3000) ;PDF파일은 스풀이 안되므로 3초 딜레이 넣음. 연속으로 파일 저장 안될 경우 시간을 늘리세요. 10초는 10000임
;;여기까지 기본 세팅


;;공통세팅
 (setq pppscaleonfit "on"); 옵션(on/fit) on은 용지에 스케일 자동 조정, fit는 용지에 맞춤
 (setq pltovrd 1); PLOTTRANSPARENCYOVERRIDE 값을 1 또는 2로 선택합니다 1일 경우 해치 투명도를 현재 캐드파일 기준으로, 2는 해치를 투명하게 바꿉니다. 플롯 속도느려짐. 기본값은 1임


;;배치탭 세팅
 (setq mmmvorder "N"); 도면공간 먼저 플롯 옵션 Y 또는 N 
 (setq mmmvhide "N"); 도면공간 객체숨기기 Y 또는 N (옵션 글자 외곽으로 보일 경우 N)


;; 여기까지 공통 세팅


;; 여기에서부터 A3 출력
 (defun c:ppp( / ent pwgs ss1  ppw1 ppw2 vpw1 vpw2 pwwsel pdgscc1 pdgscc pdgscc wlp) 
 (defun *error* (msg)(princ "error: ")(princ msg)
 (setvar "osmode" 0)
 (princ))
 (vl-load-com)
 (setq ent nil)

 (prompt "\n 프린트 할 도면을 선택하세요(블록으로된 도면외곽) !!" ) 
 (setq pwgs (ssget (list (cons 0 "insert"))))

	;; Doug C. Broad, Jr.
	;; can be used with vla-transformby to
	;; transform objects from the UCS to the WCS
	(defun UCS2WCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 1 0 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 0 1)
	)
	(list '(0 0 0 1))
	)
	)
	)
	;; transform objects from the WCS to the UCS
	(defun WCS2UCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 0 1 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 1 0)
	)
	(list '(0 0 0 1))
	)
	)
	)


  (if pwgs
   (progn
    (setq ss1 pwgs)
    (setq n 0)
    (repeat (sslength ss1)
    (setq ent (ssname ss1 n))


	(vla-TransformBy (vlax-ename->vla-object ent) (UCS2WCSMatrix))
	(vla-getboundingbox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)
	(vla-TransformBy (vlax-ename->vla-object ent) (WCS2UCSMatrix))
	;### 별명없음님 수정 UCS 설정된 도면 출력 
	;(vla-GetBoundingBox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)

  (setq ppw1 (vlax-safearray->list MinPt)) 
  (setq ppw2 (vlax-safearray->list MaxPt))

  (setq vpw1 (list (car ppw2) (cadr ppw1)));우하점
  (setq vpw2 (list (car ppw1) (cadr ppw2))) ;좌상점

  (setq wx (- (car ppw2) (car ppw1)) ) ;가로폭
  (setq wy (- (cadr ppw2) (cadr ppw1)) );높이
  (if (> wx wy) (setq wlp "L") (setq wlp "P"))

  (setq pwwsel (ssget "W" ppw1 ppw2 ));전체도면
  (setq pdgscc1 (distance ppw1 ppw2))

; lt스케일 적용하려면 왼쪽 주석 삭제하면 됨. 내가 그리는 도면에서는 필요 없어서 주석 처리함.
;  (setq pdgscc4 (* pdgscc 4 )) ;ltscale 스케일의4배
;  (setvar "ltscale"  pdgscc4) ; 치수스케일 도면축척과 동일 

  (setvar "osmode" 4335)

; 아래 한줄 추가함. 도면을 스케일 맞춰서 출력하도록 계산하기
  (setq PLOTSCALE (STRCAT "1=" (rtos (/ pdgscc1 pdgsccpaper) 2 0)))


; scale 용지맞춤 _ fit  용지맞춤, on 스케일 수정
  (if (= pppscaleonfit "fit") (setq PLOTSCALE "fit") (setq PLOTSCALE PLOTSCALE))


; PLOTTRANSPARENCYOVERRIDE 값 적용
(command "PLOTTRANSPARENCYOVERRIDE" pltovrd) 

; (command "zoom" ppw1 ppw2) ; 사용안함 어지러움

      (command "-PLOT"  
               "Y" 
               ""                   ;모형
               plotername               ;프린터이름
               papername                   ;용지 크기
               "M"                   ;mm , inch
               wlp                    ;페이지 가로 세로 판단후 L, P 선택
               "N" 
               "W"
               ppw2 ppw1
               PLOTSCALE
               "C"
               "Y"   
               ctbname
               "Y"                    ;선가중치플롯
               "A"                    ;음영플롯설정 / 표시되는 대로(A)/와이어프레임(W)/숨김(H)/뷰 스타일(V)/렌더(R)
               "N"                    ;플롯출력을 파일로
               "N"                    ;페이지설정 저장
               "Y"                    ;플롯진행
       )                     
	(command "osmode" "4335")
       (setq n (1+ n))
    )
   )
  ) 
  (command "PLOTTRANSPARENCYOVERRIDE" 1)
  (prompt "\n 출력완료!")
)



;; 여기에서부터 A3 PDF만들기
 (defun c:pdf( / ent pwgs ss1 ppw1 ppw2 vpw1 vpw2 pwwsel pdgscc1 pdgscc pdgscc4 wlp) 

 (defun *error* (msg)(princ "error: ")(princ msg)
  (setvar "osmode" 0)
 (princ))
 (vl-load-com)
 (setq ent nil)

 (prompt "\n 프린트 할 도면을 선택하세요(블록으로된 도면외곽) !!" ) 
 (setq pwgs (ssget (list (cons 0 "insert"))))

	;; Doug C. Broad, Jr.
	;; can be used with vla-transformby to
	;; transform objects from the UCS to the WCS
	(defun UCS2WCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 1 0 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 0 1)
	)
	(list '(0 0 0 1))
	)
	)
	)
	;; transform objects from the WCS to the UCS
	(defun WCS2UCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 0 1 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 1 0)
	)
	(list '(0 0 0 1))
	)
	)
	)


  (if pwgs
   (progn
    (setq ss1 pwgs)
    (setq n 0)
    (repeat (sslength ss1)
    (setq ent (ssname ss1 n))


	(vla-TransformBy (vlax-ename->vla-object ent) (UCS2WCSMatrix))
	(vla-getboundingbox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)
	(vla-TransformBy (vlax-ename->vla-object ent) (WCS2UCSMatrix))
	;### 별명없음님 수정 UCS 설정된 도면 출력 
	;(vla-GetBoundingBox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)

  (setq ppw1 (vlax-safearray->list MinPt)) 
  (setq ppw2 (vlax-safearray->list MaxPt))

  (setq vpw1 (list (car ppw2) (cadr ppw1)));우하점
  (setq vpw2 (list (car ppw1) (cadr ppw2))) ;좌상점

  (setq wx (- (car ppw2) (car ppw1)) ) ;가로폭
  (setq wy (- (cadr ppw2) (cadr ppw1)) );높이
  (if (> wx wy) (setq wlp "l") (setq wlp "p"))

  (setq pwwsel (ssget "W" ppw1 ppw2 ));전체도면
  (setq pdgscc1 (distance ppw1 ppw2))
  (setq pdgscc (/ pdgscc1 pdgsccpaper )) ; 도면척도 대각길이로 나눈값 A3기준(도면 폼 크기 틀릴 경우 여기를 고쳐주면 됨)

; lt스케일 적용하려면 왼쪽 주석 삭제하면 됨. 내가 그리는 도면에서는 필요 없어서 주석 처리함.
;  (setq pdgscc4 (* pdgscc 4 )) ;ltscale 스케일의4배
;  (setvar "ltscale"  pdgscc4) ; 치수스케일 도면축척과 동일 

  (setvar "osmode" 4335)

; 아래 한줄 추가함. 도면을 스케일 맞춰서 출력하도록 계산하기
  (setq PLOTSCALE (STRCAT "1=" (rtos (/ pdgscc1 pdgsccpaper) 2 0)))

; scale 용지맞춤 _ fit  용지맞춤, on 스케일 수정
  (if (= pppscaleonfit "fit") (setq PLOTSCALE "fit") (setq PLOTSCALE PLOTSCALE))

 (setq filetime (rtos (getvar "cdate") 2 6)) ; 이름에 붙일 시간

; PDF파일 이름은 현재 캐드파일_현재시간.pdf임
 (setq fname (getvar "dwgname"))
 (setq fname (substr fname 1 (- (strlen fname) 4)))
 (setq sffx (rtos (getvar "cdate") 2 6));;; 시간 불러오기
; (setq fname (strcat PATH fname sffx ".pdf")) 
 (setq fname (strcat "c:\\pdf\\" fname "_" sffx ".pdf")) 

; PLOTTRANSPARENCYOVERRIDE 값 적용
(command "PLOTTRANSPARENCYOVERRIDE" pltovrd) 

; (command "zoom" P_ppw1 P_ppw2) ; 사용안함 어지러움

 (command "filedia" "0"); 파일 창 안보이게, 보이게하려면 fildia 명령 후 1 입력

; scale 용지맞춤 _ fit  용지맞춤, on 스케일 수정
  (if (= pppscaleonfit "fit") (setq PLOTSCALE "fit") (setq PLOTSCALE PLOTSCALE))

; PLOTTRANSPARENCYOVERRIDE 값 적용
(command "PLOTTRANSPARENCYOVERRIDE" pltovrd) 

      (command "-PLOT"  
               "Y" 
               "model"                        
               P_plotername               ;프린터이름
               P_papername                   ;용지 크기
               "M"                   ;모형
               wlp                    ;페이지 가로 세로 판단후 L, P 선택
               "N" 
               "W"
               ppw2 ppw1
               PLOTSCALE
               "C"
               "Y"   
               ctbname
               "Y"                    ;선가중치플롯
               "A"                    ;음영플롯설정 / 표시되는 대로(A)/와이어프레임(W)/숨김(H)/뷰 스타일(V)/렌더(R)
	       fname			;파일 이름 c:\pdf 폴더 안에 현재시간.pdf로 저장
	       "Y"                    ;플롯 설정저장
               "Y"                    ;플롯진행
       )                     
	 (command "delay"		;PDF파일은 스풀이 안되므로 5초 딜레이 넣음. 파일 저장 안될 경우 시간을 늘리세요. 10초는 10000임 
	 PDFdelay
	 )
	(command "osmode" "4335")
       (setq n (1+ n))
    )
   )
  ) 
  (command "PLOTTRANSPARENCYOVERRIDE" 1)
(command "filedia" "1"); 파일 창 보이게
  (prompt "\n 출력완료!")
)



;;여기부터 배치탭 A3 출력
 (defun c:mmm( / ent pwgs ss1  ppw1 ppw2 vpw1 vpw2 pwwsel pdgscc1 pdgscc pdgscc4 ) 
 (defun *error* (msg)(princ "error: ")(princ msg)
 (setvar "osmode" 0)
 (princ))
 (vl-load-com)
 (setq ent nil)

 (prompt "\n 프린트 할 도면을 선택하세요(블록으로된 도면외곽) !!" ) 
 (setq pwgs (ssget (list (cons 0 "insert"))))

	;; Doug C. Broad, Jr.
	;; can be used with vla-transformby to
	;; transform objects from the UCS to the WCS
	(defun UCS2WCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 1 0 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 0 1)
	)
	(list '(0 0 0 1))
	)
	)
	)
	;; transform objects from the WCS to the UCS
	(defun WCS2UCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 0 1 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 1 0)
	)
	(list '(0 0 0 1))
	)
	)
	)


  (if pwgs
   (progn
    (setq ss1 pwgs)
    (setq n 0)
    (repeat (sslength ss1)
    (setq ent (ssname ss1 n))


	(vla-TransformBy (vlax-ename->vla-object ent) (UCS2WCSMatrix))
	(vla-getboundingbox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)
	(vla-TransformBy (vlax-ename->vla-object ent) (WCS2UCSMatrix))
	;### 별명없음님 수정 UCS 설정된 도면 출력 
	;(vla-GetBoundingBox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)

  (setq ppw1 (vlax-safearray->list MinPt)) 
  (setq ppw2 (vlax-safearray->list MaxPt))

  (setq vpw1 (list (car ppw2) (cadr ppw1)));우하점
  (setq vpw2 (list (car ppw1) (cadr ppw2))) ;좌상점

  (setq wx (- (car ppw2) (car ppw1)) ) ;가로폭
  (setq wy (- (cadr ppw2) (cadr ppw1)) );높이
  (if (> wx wy) (setq wlp "l") (setq wlp "p"))

  (setq pwwsel (ssget "W" ppw1 ppw2 ));전체도면
  (setq pdgscc1 (distance ppw1 ppw2))

; lt스케일 적용하려면 왼쪽 주석 삭제하면 됨. 내가 그리는 도면에서는 필요 없어서 주석 처리함.
;  (setq pdgscc4 (* pdgscc 4 )) ;ltscale 스케일의4배
;  (setvar "ltscale"  pdgscc4) ; 치수스케일 도면축척과 동일 

  (setvar "osmode" 4335)

; 아래 한줄 추가함. 도면을 스케일 맞춰서 출력하도록 계산하기
  (setq PLOTSCALE (STRCAT "1=" (rtos (/ pdgscc1 pdgsccpaper) 2 0)))

; scale 용지맞춤 _ fit  용지맞춤, on 스케일 수정
  (if (= pppscaleonfit "fit") (setq PLOTSCALE "fit") (setq PLOTSCALE PLOTSCALE))

; PLOTTRANSPARENCYOVERRIDE 값 적용
(command "PLOTTRANSPARENCYOVERRIDE" pltovrd) 

; (command "zoom" ppw1 ppw2) ; 사용안함 어지러움

      (command "-PLOT"  
               "Y" 
               ""                        
               plotername               ;프린터이름
               papername                   ;용지 크기
               "M"                   ;밀리미터 인치
               wlp                    ;페이지 가로 세로 판단후 L, P 선택
               "N" 
               "W"
               ppw2 ppw1
               PLOTSCALE
               "C"
               "Y"   
               ctbname
               "Y"                    ;선가중치플롯
               "N"                    ;선가중치플롯 축척에 적용
	       mmmvorder              ;도면공간먼저플롯
               mmmvhide               ;도면공간객체숨기기 20200326 N으로 변경 (글자 외곽만 보여서 수정함)
	       "N"                    ;플롯 파일저장
	       "N"                    ;플롯 설정저장
               "Y"                    ;플롯진행
       )                     
	(command "osmode" "4335")
       (setq n (1+ n))
    )
   )
  ) 
  (command "PLOTTRANSPARENCYOVERRIDE" 1)
  (prompt "\n 출력완료!")
)



;; 여기에서부터 배치탭 A3 PDF만들기
 (defun c:pdfm( / ent pwgs ss1 ppw1 ppw2 vpw1 vpw2 pwwsel pdgscc1 pdgscc pdgscc4 wlp) 

 (defun *error* (msg)(princ "error: ")(princ msg)
  (setvar "osmode" 0)
 (princ))
 (vl-load-com)
 (setq ent nil)

 (prompt "\n 프린트 할 도면을 선택하세요(블록으로된 도면외곽) !!" ) 
 (setq pwgs (ssget (list (cons 0 "insert"))))

	;; Doug C. Broad, Jr.
	;; can be used with vla-transformby to
	;; transform objects from the UCS to the WCS
	(defun UCS2WCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 1 0 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 0 1)
	)
	(list '(0 0 0 1))
	)
	)
	)
	;; transform objects from the WCS to the UCS
	(defun WCS2UCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 0 1 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 1 0)
	)
	(list '(0 0 0 1))
	)
	)
	)


  (if pwgs
   (progn
    (setq ss1 pwgs)
    (setq n 0)
    (repeat (sslength ss1)
    (setq ent (ssname ss1 n))


	(vla-TransformBy (vlax-ename->vla-object ent) (UCS2WCSMatrix))
	(vla-getboundingbox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)
	(vla-TransformBy (vlax-ename->vla-object ent) (WCS2UCSMatrix))
	;### 별명없음님 수정 UCS 설정된 도면 출력 
	;(vla-GetBoundingBox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)

  (setq ppw1 (vlax-safearray->list MinPt)) 
  (setq ppw2 (vlax-safearray->list MaxPt))

  (setq vpw1 (list (car ppw2) (cadr ppw1)));우하점
  (setq vpw2 (list (car ppw1) (cadr ppw2))) ;좌상점

  (setq wx (- (car ppw2) (car ppw1)) ) ;가로폭
  (setq wy (- (cadr ppw2) (cadr ppw1)) );높이
  (if (> wx wy) (setq wlp "l") (setq wlp "p"))

  (setq pwwsel (ssget "W" ppw1 ppw2 ));전체도면
  (setq pdgscc1 (distance ppw1 ppw2))
  (setq pdgscc (/ pdgscc1 pdgsccpaper )) ; 도면척도 대각길이로 나눈값 A3기준(도면 폼 크기 틀릴 경우 여기를 고쳐주면 됨)

; lt스케일 적용하려면 왼쪽 주석 삭제하면 됨. 내가 그리는 도면에서는 필요 없어서 주석 처리함.
;  (setq pdgscc4 (* pdgscc 4 )) ;ltscale 스케일의4배
;  (setvar "ltscale"  pdgscc4) ; 치수스케일 도면축척과 동일 

  (setvar "osmode" 4335)

; 아래 한줄 추가함. 도면을 스케일 맞춰서 출력하도록 계산하기
  (setq PLOTSCALE (STRCAT "1=" (rtos (/ pdgscc1 pdgsccpaper) 2 0)))

; scale 용지맞춤 _ fit  용지맞춤, on 스케일 수정
  (if (= pppscaleonfit "fit") (setq PLOTSCALE "fit") (setq PLOTSCALE PLOTSCALE))

 (setq filetime (rtos (getvar "cdate") 2 6)) ; 이름에 붙일 시간

; PDF파일 이름은 현재 캐드파일_현재시간.pdf임
 (setq fname (getvar "dwgname"))
 (setq fname (substr fname 1 (- (strlen fname) 4)))
 (setq sffx (rtos (getvar "cdate") 2 6));;; 시간 불러오기
; (setq fname (strcat PATH fname sffx ".pdf")) 
 (setq fname (strcat "c:\\pdf\\" fname "_" sffx ".pdf")) 


; PLOTTRANSPARENCYOVERRIDE 값 적용
(command "PLOTTRANSPARENCYOVERRIDE" pltovrd) 

; (command "zoom" P_ppw1 P_ppw2) ; 사용안함 어지러움

 (command "filedia" "0"); 파일 창 안보이게, 보이게하려면 fildia 명령 후 1 입력
 
      (command "-PLOT"  
               "Y" 
               ""                         ;배치모형 현재로
               P_plotername               ;프린터이름
               P_papername                   ;용지 크기
               "M"                   ;밀리미터
               wlp                    ;페이지 가로 세로 판단후 L, P 선택
               "N" 
               "W"
               ppw2 ppw1
               PLOTSCALE
               "C"
               "Y"   
               ctbname
               "Y"                    ;선가중치플롯
               "N"                    ;선가중치플롯 축척에 적용
	       mmmvorder              ;도면공간먼저플롯
               mmmvhide               ;도면공간객체숨기기 20200326 N으로 변경 (글자 외곽만 보여서 수정함)
	       fname			;파일 이름 c:\pdf 폴더 안에 현재시간.pdf로 저장
	       "N"                    ;플롯 설정저장
               "Y"                    ;플롯진행
       )                     
	 (command "delay"		;PDF파일은 스풀이 안되므로 5초 딜레이 넣음. 파일 저장 안될 경우 시간을 늘리세요. 10초는 10000임 
	 PDFdelay
	 )
	(command "osmode" "4335")
       (setq n (1+ n))
    )
   )
  ) 
  (command "PLOTTRANSPARENCYOVERRIDE" 1)
(command "filedia" "1"); 파일 창 보이게
  (prompt "\n 출력완료!")
)






;; 아래부터는 A4용지 출력 되도록 추가 했습니다. 추가명령 ppp4, pdf4, mmmm4, pdfm4)

;; 여기에서부터 A4 프린터 출력
 (defun c:ppp4( / ent pwgs ss1  ppw1 ppw2 vpw1 vpw2 pwwsel pdgscc1 pdgscc pdgscc4 ) 
 (defun *error* (msg)(princ "error: ")(princ msg)
 (setvar "osmode" 0)
 (princ))
 (vl-load-com)
 (setq ent nil)

 (prompt "\n 프린트 할 도면을 선택하세요(블록으로된 도면외곽) !!" ) 
 (setq pwgs (ssget (list (cons 0 "insert"))))

	;; Doug C. Broad, Jr.
	;; can be used with vla-transformby to
	;; transform objects from the UCS to the WCS
	(defun UCS2WCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 1 0 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 0 1)
	)
	(list '(0 0 0 1))
	)
	)
	)
	;; transform objects from the WCS to the UCS
	(defun WCS2UCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 0 1 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 1 0)
	)
	(list '(0 0 0 1))
	)
	)
	)


  (if pwgs
   (progn
    (setq ss1 pwgs)
    (setq n 0)
    (repeat (sslength ss1)
    (setq ent (ssname ss1 n))


	(vla-TransformBy (vlax-ename->vla-object ent) (UCS2WCSMatrix))
	(vla-getboundingbox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)
	(vla-TransformBy (vlax-ename->vla-object ent) (WCS2UCSMatrix))
	;### 별명없음님 수정 UCS 설정된 도면 출력 
	;(vla-GetBoundingBox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)

  (setq ppw1 (vlax-safearray->list MinPt)) 
  (setq ppw2 (vlax-safearray->list MaxPt))

  (setq vpw1 (list (car ppw2) (cadr ppw1)));우하점
  (setq vpw2 (list (car ppw1) (cadr ppw2))) ;좌상점

  (setq wx (- (car ppw2) (car ppw1)) ) ;가로폭
  (setq wy (- (cadr ppw2) (cadr ppw1)) );높이
  (if (> wx wy) (setq wlpa4 "L") (setq wlpa4 "P"))

  (setq pwwsel (ssget "W" ppw1 ppw2 ));전체도면
  (setq pdgscc1 (distance ppw1 ppw2))

; lt스케일 적용하려면 왼쪽 주석 삭제하면 됨. 내가 그리는 도면에서는 필요 없어서 주석 처리함.
;  (setq pdgscc4 (* pdgscc 4 )) ;ltscale 스케일의4배
;  (setvar "ltscale"  pdgscc4) ; 치수스케일 도면축척과 동일 

  (setvar "osmode" 4335)

; 아래 한줄 추가함. 도면을 스케일 맞춰서 출력하도록 계산하기
  (setq PLOTSCALEA4 (STRCAT "1=" (rtos (/ pdgscc1 pdgsccpapera4) 2 0)))


; scale 용지맞춤 _ fit  용지맞춤, on 스케일 수정
  (if (= pppscaleonfit "fit") (setq PLOTSCALEA4 "fit") (setq PLOTSCALEA4 PLOTSCALEA4))


; PLOTTRANSPARENCYOVERRIDE 값 적용
(command "PLOTTRANSPARENCYOVERRIDE" pltovrd) 

; (command "zoom" ppw1 ppw2) ; 사용안함 어지러움

      (command "-PLOT"  
               "Y" 
               ""                        
               plotername               ;프린터이름
               papernamea4                   ;용지 크기
               "M"                   ;모형
               wlpa4                    ;페이지 가로 세로 판단후 L, P 선택
               "N" 
               "W"
               ppw2 ppw1
               PLOTSCALEA4
               "C"
               "Y"   
               ctbnameA4
               "Y"                    ;선가중치플롯
               "A"                    ;음영플롯설정 / 표시되는 대로(A)/와이어프레임(W)/숨김(H)/뷰 스타일(V)/렌더(R)
               "N"                    ;플롯출력을 파일로
               "N"                    ;페이지설정 저장
               "Y"                    ;플롯진행
       )                     
	(command "osmode" "4335")
       (setq n (1+ n))
    )
   )
  ) 
  (command "PLOTTRANSPARENCYOVERRIDE" 1)

  (prompt "\n 출력완료!")
)



;; 여기에서부터 A4 PDF만들기
 (defun c:pdf4( / ent pwgs ss1 ppw1 ppw2 vpw1 vpw2 pwwsel pdgscc1 pdgscc pdgscc4 wlpa4) 

 (defun *error* (msg)(princ "error: ")(princ msg)
  (setvar "osmode" 0)
 (princ))
 (vl-load-com)
 (setq ent nil)

 (prompt "\n 프린트 할 도면을 선택하세요(블록으로된 도면외곽) !!" ) 
 (setq pwgs (ssget (list (cons 0 "insert"))))

	;; Doug C. Broad, Jr.
	;; can be used with vla-transformby to
	;; transform objects from the UCS to the WCS
	(defun UCS2WCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 1 0 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 0 1)
	)
	(list '(0 0 0 1))
	)
	)
	)
	;; transform objects from the WCS to the UCS
	(defun WCS2UCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 0 1 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 1 0)
	)
	(list '(0 0 0 1))
	)
	)
	)


  (if pwgs
   (progn
    (setq ss1 pwgs)
    (setq n 0)
    (repeat (sslength ss1)
    (setq ent (ssname ss1 n))


	(vla-TransformBy (vlax-ename->vla-object ent) (UCS2WCSMatrix))
	(vla-getboundingbox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)
	(vla-TransformBy (vlax-ename->vla-object ent) (WCS2UCSMatrix))
	;### 별명없음님 수정 UCS 설정된 도면 출력 
	;(vla-GetBoundingBox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)

  (setq ppw1 (vlax-safearray->list MinPt)) 
  (setq ppw2 (vlax-safearray->list MaxPt))

  (setq vpw1 (list (car ppw2) (cadr ppw1)));우하점
  (setq vpw2 (list (car ppw1) (cadr ppw2))) ;좌상점

  (setq wx (- (car ppw2) (car ppw1)) ) ;가로폭
  (setq wy (- (cadr ppw2) (cadr ppw1)) );높이
  (if (> wx wy) (setq wlpa4 "l") (setq wlpa4 "p"))

  (setq pwwsel (ssget "W" ppw1 ppw2 ));전체도면
  (setq pdgscc1 (distance ppw1 ppw2))
  (setq pdgscc (/ pdgscc1 pdgsccpapera4 )) ; 도면척도 대각길이로 나눈값 A3기준(도면 폼 크기 틀릴 경우 여기를 고쳐주면 됨)

; lt스케일 적용하려면 왼쪽 주석 삭제하면 됨. 내가 그리는 도면에서는 필요 없어서 주석 처리함.
;  (setq pdgscc4 (* pdgscc 4 )) ;ltscale 스케일의4배
;  (setvar "ltscale"  pdgscc4) ; 치수스케일 도면축척과 동일 

  (setvar "osmode" 4335)

; 아래 한줄 추가함. 도면을 스케일 맞춰서 출력하도록 계산하기
  (setq PLOTSCALEA4 (STRCAT "1=" (rtos (/ pdgscc1 pdgsccpapera4) 2 0)))

; scale 용지맞춤 _ fit  용지맞춤, on 스케일 수정
  (if (= pppscaleonfit "fit") (setq PLOTSCALEA4 "fit") (setq PLOTSCALEA4 PLOTSCALEA4))

 (setq filetime (rtos (getvar "cdate") 2 6)) ; 이름에 붙일 시간

; PDF파일 이름은 현재 캐드파일_현재시간.pdf임
 (setq fname (getvar "dwgname"))
 (setq fname (substr fname 1 (- (strlen fname) 4)))
 (setq sffx (rtos (getvar "cdate") 2 6));;; 시간 불러오기
; (setq fname (strcat PATH fname sffx ".pdf")) 
 (setq fname (strcat "c:\\pdf\\" fname "_" sffx ".pdf")) 


; PLOTTRANSPARENCYOVERRIDE 값 적용
(command "PLOTTRANSPARENCYOVERRIDE" pltovrd) 

; (command "zoom" P_ppw1 P_ppw2) ; 사용안함 어지러움

 (command "filedia" "0"); 파일 창 안보이게, 보이게하려면 fildia 명령 후 1 입력

      (command "-PLOT"  
               "Y" 
               "model"                        
               P_plotername               ;프린터이름
               P_papernamea4                   ;용지 크기
               "M"                   ;모형
               wlpa4                    ;페이지 가로 세로 판단후 L, P 선택
               "N" 
               "W"
               ppw2 ppw1
               PLOTSCALEA4
               "C"
               "Y"   
               ctbnameA4
               "Y"                    ;선가중치플롯
               "A"                    ;음영플롯설정 / 표시되는 대로(A)/와이어프레임(W)/숨김(H)/뷰 스타일(V)/렌더(R)
	       fname			;파일 이름 c:\pdf 폴더 안에 현재시간.pdf로 저장
	       "Y"                    ;플롯 설정저장
               "Y"                    ;플롯진행
       )                     
	 (command "delay"		;PDF파일은 스풀이 안되므로 5초 딜레이 넣음. 파일 저장 안될 경우 시간을 늘리세요. 10초는 10000임 
	 PDFdelay
	 )
	(command "osmode" "4335")
       (setq n (1+ n))
    )
   )
  ) 
  (command "PLOTTRANSPARENCYOVERRIDE" 1)
  (command "filedia" "1"); 파일 창 보이게
  (prompt "\n 출력완료!")
)



;;여기부터 배치탭 A4 출력
 (defun c:mmm4( / ent pwgs ss1  ppw1 ppw2 vpw1 vpw2 pwwsel pdgscc1 pdgscc pdgscc4 ) 
 (defun *error* (msg)(princ "error: ")(princ msg)
 (setvar "osmode" 0)
 (princ))
 (vl-load-com)
 (setq ent nil)

 (prompt "\n 프린트 할 도면을 선택하세요(블록으로된 도면외곽) !!" ) 
 (setq pwgs (ssget (list (cons 0 "insert"))))

	;; Doug C. Broad, Jr.
	;; can be used with vla-transformby to
	;; transform objects from the UCS to the WCS
	(defun UCS2WCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 1 0 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 0 1)
	)
	(list '(0 0 0 1))
	)
	)
	)
	;; transform objects from the WCS to the UCS
	(defun WCS2UCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 0 1 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 1 0)
	)
	(list '(0 0 0 1))
	)
	)
	)


  (if pwgs
   (progn
    (setq ss1 pwgs)
    (setq n 0)
    (repeat (sslength ss1)
    (setq ent (ssname ss1 n))


	(vla-TransformBy (vlax-ename->vla-object ent) (UCS2WCSMatrix))
	(vla-getboundingbox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)
	(vla-TransformBy (vlax-ename->vla-object ent) (WCS2UCSMatrix))
	;### 별명없음님 수정 UCS 설정된 도면 출력 
	;(vla-GetBoundingBox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)

  (setq ppw1 (vlax-safearray->list MinPt)) 
  (setq ppw2 (vlax-safearray->list MaxPt))

  (setq vpw1 (list (car ppw2) (cadr ppw1)));우하점
  (setq vpw2 (list (car ppw1) (cadr ppw2))) ;좌상점

  (setq wx (- (car ppw2) (car ppw1)) ) ;가로폭
  (setq wy (- (cadr ppw2) (cadr ppw1)) );높이
  (if (> wx wy) (setq wlpa4 "l") (setq wlpa4 "p"))

  (setq pwwsel (ssget "W" ppw1 ppw2 ));전체도면
  (setq pdgscc1 (distance ppw1 ppw2))

; lt스케일 적용하려면 왼쪽 주석 삭제하면 됨. 내가 그리는 도면에서는 필요 없어서 주석 처리함.
;  (setq pdgscc4 (* pdgscc 4 )) ;ltscale 스케일의4배
;  (setvar "ltscale"  pdgscc4) ; 치수스케일 도면축척과 동일 

  (setvar "osmode" 4335)

; 아래 한줄 추가함. 도면을 스케일 맞춰서 출력하도록 계산하기
  (setq PLOTSCALEA4 (STRCAT "1=" (rtos (/ pdgscc1 pdgsccpapera4) 2 0)))

; scale 용지맞춤 _ fit  용지맞춤, on 스케일 수정
  (if (= pppscaleonfit "fit") (setq PLOTSCALEA4 "fit") (setq PLOTSCALEA4 PLOTSCALEA4))

; PLOTTRANSPARENCYOVERRIDE 값 적용
(command "PLOTTRANSPARENCYOVERRIDE" pltovrd) 

; (command "zoom" ppw1 ppw2) ; 사용안함 어지러움

      (command "-PLOT"  
               "Y" 
               ""                        
               plotername               ;프린터이름
               papernamea4                   ;용지 크기
               "M"                   ;밀리미터 인치
               wlpa4                    ;페이지 가로 세로 판단후 L, P 선택
               "N" 
               "W"
               ppw2 ppw1
               PLOTSCALEA4
               "C"
               "Y"   
               ctbnameA4
               "Y"                    ;선가중치플롯
               "N"                    ;선가중치플롯 축척에 적용
	       mmmvorder              ;도면공간먼저플롯
               mmmvhide               ;도면공간객체숨기기 20200326 N으로 변경 (글자 외곽만 보여서 수정함)
	       "N"                    ;플롯 파일저장
	       "N"                    ;플롯 설정저장
               "Y"                    ;플롯진행
       )                     
	(command "osmode" "4335")
       (setq n (1+ n))
    )
   )
  ) 
  (command "PLOTTRANSPARENCYOVERRIDE" 1)
  (prompt "\n 출력완료!")
)



;; 여기에서부터 배치탭 A4 PDF만들기
 
 ;;아래는 PDF로 출력하는 용도의 세팅입니다. CTB는 종이 출력 용도와 동일하게 따라갑니다.
; (setq P_plotername "DWG to PDF.pc5");PDF출력할 플롯터 이름 (Adobe PDF.pc3은 파일 저장 문제로 잘 안됩니다.)
; (setq P_papername "ISO expand A3 (420.00 x 297.00 MM)") ; 용지 크기는 A3 종이 종류 선택, 프린터마다 상태 다르니 확인바람. 
; (setq P_papernamea4 "ISO expand A4 (297.00 x 210.00 MM)") ; 용지 크기는 A3 종이 종류 선택, 프린터마다 상태 다르니 확인바람. 
			 ;예) ISO 전체페이지 A3(420.00 x 297.00 mm) 등으로 설정됨
; (setq PATH (getvar "c:\\pdf\\"))	;;; pdf파일 저장 위치 현재는 c:\pdf 폴더임. 폴더 없을 경우는 만드세요.
; (setq PDFdelay 3000) ;PDF파일은 스풀이 안되므로 3초 딜레이 넣음. 연속으로 파일 저장 안될 경우 시간을 늘리세요. 10초는 10000임

 (defun c:pdfm4( / ent pwgs ss1 ppw1 ppw2 vpw1 vpw2 pwwsel pdgscc1 pdgscc pdgscc4 wlpa4) 

 (defun *error* (msg)(princ "error: ")(princ msg)
  (setvar "osmode" 0)
 (princ))
 (vl-load-com)
 (setq ent nil)

 (prompt "\n 프린트 할 도면을 선택하세요(블록으로된 도면외곽) !!" ) 
 (setq pwgs (ssget (list (cons 0 "insert"))))

	;; Doug C. Broad, Jr.
	;; can be used with vla-transformby to
	;; transform objects from the UCS to the WCS
	(defun UCS2WCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 1 0 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 0 1)
	)
	(list '(0 0 0 1))
	)
	)
	)
	;; transform objects from the WCS to the UCS
	(defun WCS2UCSMatrix ()
	(vlax-tmatrix
	(append
	(mapcar
	'(lambda (vector origin)
	(append (trans vector 0 1 t) (list origin))
	)
	(list '(1 0 0) '(0 1 0) '(0 0 1))
	(trans '(0 0 0) 1 0)
	)
	(list '(0 0 0 1))
	)
	)
	)


  (if pwgs
   (progn
    (setq ss1 pwgs)
    (setq n 0)
    (repeat (sslength ss1)
    (setq ent (ssname ss1 n))


	(vla-TransformBy (vlax-ename->vla-object ent) (UCS2WCSMatrix))
	(vla-getboundingbox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)
	(vla-TransformBy (vlax-ename->vla-object ent) (WCS2UCSMatrix))
	;### 별명없음님 수정 UCS 설정된 도면 출력 
	;(vla-GetBoundingBox (vlax-ename->vla-object ent) 'MinPt 'MaxPt)

  (setq ppw1 (vlax-safearray->list MinPt)) 
  (setq ppw2 (vlax-safearray->list MaxPt))

  (setq vpw1 (list (car ppw2) (cadr ppw1)));우하점
  (setq vpw2 (list (car ppw1) (cadr ppw2))) ;좌상점

  (setq wx (- (car ppw2) (car ppw1)) ) ;가로폭
  (setq wy (- (cadr ppw2) (cadr ppw1)) );높이
  (if (> wx wy) (setq wlpa4 "l") (setq wlpa4 "p"))

  (setq pwwsel (ssget "W" ppw1 ppw2 ));전체도면
  (setq pdgscc1 (distance ppw1 ppw2))
  (setq pdgscc (/ pdgscc1 pdgsccpapera4 )) ; 도면척도 대각길이로 나눈값 A3기준(도면 폼 크기 틀릴 경우 여기를 고쳐주면 됨)

; lt스케일 적용하려면 왼쪽 주석 삭제하면 됨. 내가 그리는 도면에서는 필요 없어서 주석 처리함.
;  (setq pdgscc4 (* pdgscc 4 )) ;ltscale 스케일의4배
;  (setvar "ltscale"  pdgscc4) ; 치수스케일 도면축척과 동일 

  (setvar "osmode" 4335)

; 아래 한줄 추가함. 도면을 스케일 맞춰서 출력하도록 계산하기
  (setq PLOTSCALEA4 (STRCAT "1=" (rtos (/ pdgscc1 pdgsccpapera4) 2 0)))

; scale 용지맞춤 _ fit  용지맞춤, on 스케일 수정
  (if (= pppscaleonfit "fit") (setq PLOTSCALEA4 "fit") (setq PLOTSCALEA4 PLOTSCALEA4))

 (setq filetime (rtos (getvar "cdate") 2 6)) ; 이름에 붙일 시간

; PDF파일 이름은 현재 캐드파일_현재시간.pdf임
 (setq fname (getvar "dwgname"))
 (setq fname (substr fname 1 (- (strlen fname) 4)))
 (setq sffx (rtos (getvar "cdate") 2 6));;; 시간 불러오기
; (setq fname (strcat PATH fname sffx ".pdf")) 
 (setq fname (strcat "c:\\pdf\\" fname "_" sffx ".pdf")) 


; PLOTTRANSPARENCYOVERRIDE 값 적용
(command "PLOTTRANSPARENCYOVERRIDE" pltovrd) 

; (command "zoom" P_ppw1 P_ppw2) ; 사용안함 어지러움

 (command "filedia" "0"); 파일 창 안보이게, 보이게하려면 fildia 명령 후 1 입력

      (command "-PLOT"  
               "Y" 
               ""                         ;배치모형 현재로
               P_plotername               ;프린터이름
               P_papernamea4                   ;용지 크기
               "M"                   ;밀리미터
               wlpa4                    ;페이지 가로 세로 판단후 L, P 선택
               "N" 
               "W"
               ppw2 ppw1
               PLOTSCALEA4
               "C"
               "Y"   
               ctbnameA4
               "Y"                    ;선가중치플롯
               "N"                    ;선가중치플롯 축척에 적용
	       mmmvorder              ;도면공간먼저플롯
               mmmvhide               ;도면공간객체숨기기 20200326 N으로 변경 (글자 외곽만 보여서 수정함)
	       fname			;파일 이름 c:\pdf 폴더 안에 현재시간.pdf로 저장
	       "N"                    ;플롯 설정저장
               "Y"                    ;플롯진행
       )                     
	 (command "delay"		;PDF파일은 스풀이 안되므로 5초 딜레이 넣음. 파일 저장 안될 경우 시간을 늘리세요. 10초는 10000임 
	 PDFdelay
	 )
	(command "osmode" "4335")
       (setq n (1+ n))
    )
   )
  ) 
  (command "PLOTTRANSPARENCYOVERRIDE" 1)
(command "filedia" "1"); 파일 창 보이게
  (prompt "\n 출력완료!")
)



;;===========================================================
; 출력하기 메뉴 2023.12.28. Kim Hyun Chul (kimhyunchul.co.kr)
;  ->선택한 도각을 출력하는 리습
;;------ PPP_zw.dcl / PPP_autocad_Menu / PPP_autocad_Plot --------------------------------
(defun c:pppp(/ dlgid dname lno ra1 ra2 ra3 ra4 ra5 ra6 ra7 ra8 ra9 tem)
   (setq dlgid (load_dialog "c:/PDF/PPP/PPP_zw.dcl"))
   (setq dname "ppp_dcl_gstar")
   (if (not (new_dialog dname dlgid)) (exit))
   (setq lno nil tem 3)
   (set_tile "ra1" "1")
   (action_tile "ra1" "(setq lno $key)")
   (action_tile "ra2" "(setq lno $key)")
   (action_tile "ra3" "(setq lno $key)")
   (action_tile "ra4" "(setq lno $key)")
   (action_tile "ra5" "(setq lno $key)")
   (action_tile "ra6" "(setq lno $key)")
   (action_tile "ra7" "(setq lno $key)")
   (action_tile "ra8" "(setq lno $key)")
   (action_tile "ra9" "(setq lno $key)")
   (action_tile "accept"  "(setq tem 9)(done_dialog)")
   (start_dialog)
   (unload_dialog dlgid)
   (if (= tem 9) (progn
      (if (= lno nil) (setq lno "ra1"))
      (cond
         ((= lno "ra1") (c:ppp))
         ((= lno "ra2") (c:pdf))
         ((= lno "ra3") (c:ppp4))
         ((= lno "ra4") (c:pdf4))
         ((= lno "ra5") (c:mmm))
         ((= lno "ra6") (c:pdfm))
         ((= lno "ra7") (c:mmm4))
         ((= lno "ra8") (c:pdfm4))
         ((= lno "ra9") (vl-cmdf "zwplot"))
      )
   ))
(prin1))
