(defun c:HelloWorld ()
 (command "._text" "0,0" 1.5 0 "Hello, World!") ; (0,0) 위치에 높이가 2.5이고 각도가 0인 텍스트 생성
 (princ)
)
(c:HelloWorld)
​​