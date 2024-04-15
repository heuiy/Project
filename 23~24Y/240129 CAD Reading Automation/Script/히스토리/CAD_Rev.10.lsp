;; 다시 시작!!!

(defun c:ExportAreaToPDF (/ doc layout p1 p2 variantArray)
 (setq doc (vla-get-ActiveDocument (vlax-get-acad-object))
       layout (vla-get-ActiveLayout doc)
 )
​
;; 좌표 설정 함수
(defun SetPlotArea (x1 y1 x2 y2)
 (setq p1 (vlax-make-safearray vlax-vbDouble '(0 1)) ; 두 요소를 가진 배열 생성
       p2 (vlax-make-safearray vlax-vbDouble '(0 1)) ; 두 요소를 가진 배열 생성
 )
 (vlax-safearray-fill p1 (list x1 y1))
 (vlax-safearray-fill p2 (list x2 y2))
 (setq p1 (vlax-make-variant p1)
       p2 (vlax-make-variant p2)
 )
 (vla-put-PlotWindowArea layout (vlax-make-safearray vlax-vbVariant (list p1 p2)))
 (vla-PutPlotType layout 2) ; 2 = Window
)
​​
;; PDF 출력 함수
(defun PrintPDF (filename)
  (vla-SetPlotConfigurationName layout "DWG To PDF.pc3" "ISO_full_bleed_A3_(420.00_x_297.00_MM)")
  (vla-PutPlotRotation layout 0)
  (vla-PutPlotCentered layout :vlax-true)
  (vla-PutPlotWithLineweights layout :vlax-true)
  (vla-Plot layout) ; Plot 메서드 호출
  (vla-PlotToFile layout filename)
  (princ (strcat "\nExported to " filename))
)
​
;; 첫 번째 영역 출력
(SetPlotArea -3183.1443 751.1082 -4058.7059 143.0281)
(PrintPDF "D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\240129 CAD Reading Automation\\PDF\\in\\Plot1.pdf")
​
;; 두 번째 영역 출력
(SetPlotArea -4051.286 743.6926 -4926.8476 128.1969)
(PrintPDF "D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\240129 CAD Reading Automation\\PDF\\in\\Plot2.pdf")
​
;; 세 번째 영역 출력
(SetPlotArea -4949.1077 743.6926 -5794.9893 135.6125)
(PrintPDF "D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\240129 CAD Reading Automation\\PDF\\in\\Plot3.pdf")
​
(princ)
)
​​​