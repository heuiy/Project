(defun c:DrawCircle ()
 (command "circle" "0,0" 11) ; 중심점 (0,0)에 반지름이 10인 원을 그립니다.
 (princ "circle is created") ; 사용자에게 원이 생성되었다고 알립니다.
 (princ)
)