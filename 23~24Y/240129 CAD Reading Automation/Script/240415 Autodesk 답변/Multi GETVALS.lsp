; Getvals multi allows multiple line inputs
; By Alan H Feb 2019 info@alanh.com.au
; 
; code examples
; the input box size can be bigger just change the two values tetsed with 145 145 and worked ok.
; (if (not AH:getvalsm)(load "Multi Getvals.lsp"))
; (setq ans (AH:getvalsm (list "Enter values " "Length " 5 4 "6" "Dia " 5 4 "1" "Offset " 5 4 "6" "Dia " 5 4 "3/4")))
; (setq ans (AH:getvalsm (list "Enter values " "Length " 5 4 "6" "width" 5 4 "1")))
; (setq l1 (atof (car ans)) l2 (atof (cadr ans)))
;
; (setq L1 (atof (nth 0 ans)) L2 (atof (nth 1 ans)) L3 (atof (nth 2 ans)) L4 (atof (nth 3 ans)))

; note the values are strings not numbers can be any number including decimal "25.4" 

; (setq ans (AH:getvalsm (list "Enter Values" "Length      " 5 4 "100" "Width" 5 4 "50" "Depth" 5 4 "25" "Gap" 5 4 "25")))
; note the values are strings so use atof to convert to real number others can be strings
; (setq len (atof (nth 0 ans)) wid (atof (nth 1 ans)) depth (atof (nth 2 ans)) gap (atof (nth 3 ans)))

; mix of numbers and strings
; (setq ans (AH:getvalsm (list "Enter Values" "Length      " 5 4 "100" "Width" 5 4 "50" "Depth" 5 4 "25" "Gap type" 5 4 "A")))
; (setq len (atof (nth 0 ans)) wid (atof (nth 1 ans)) depth (atof (nth 2 ans)) gap (nth 3 ans))

(defun AH:getvalsm (dcllst / x y num fo fname keynum key_lst v_lst)
  (setq num (/ (- (length dcllst) 1) 4))
  (setq x 0)
  (setq y 0)
  (setq fo (open (setq fname (vl-filename-mktemp "" "" ".dcl")) "w"))
  (write-line "ddgetvalAH : dialog {" fo)
  (write-line (strcat "	label =" (chr 34) (nth 0 dcllst) (chr 34) " ;") fo)
  (write-line " : column {" fo)
  (write-line " width =25;" fo)
  (repeat num
    (write-line "spacer_1 ;" fo)
    (write-line ": edit_box {" fo)
    (setq keynum (strcat "key" (rtos (setq y (+ Y 1)) 2 0)))
    (write-line (strcat "    key = " (chr 34) keynum (chr 34) ";") fo)
    (write-line (strcat " label = " (chr 34) (nth (+ x 1) dcllst) (chr 34) ";") fo)
    (write-line (strcat "     edit_width = " (rtos (nth (+ x 2) dcllst) 2 0) ";") fo)
    (write-line (strcat "     edit_limit = " (rtos (nth (+ x 3) dcllst) 2 0) ";") fo)
    (write-line "   is_enabled = true ;" fo)
    (write-line "   allow_accept=true ;" fo)
    (write-line "    }" fo)
    (setq x (+ x 4))
  )
  (write-line "    }" fo)
  (write-line "spacer_1 ;" fo)
  (write-line "ok_cancel;}" fo)
  (close fo)

  (setq dcl_id (load_dialog fname))
  (if (not (new_dialog "ddgetvalAH" dcl_id))
    (exit)
  )
  (setq x 0)
  (setq y 0)
  (setq v_lst '())
  (repeat num
    (setq keynum (strcat "key" (rtos (setq y (+ Y 1)) 2 0)))
    (setq key_lst (cons keynum key_lst))
    (set_tile keynum (nth (setq x (+ x 4)) dcllst))
   ; (mode_tile keynum 3)
  )
    (mode_tile "key1" 2)
  (action_tile "accept" "(mapcar '(lambda (x) (setq v_lst (cons (get_tile x) v_lst))) key_lst)(done_dialog)")
  (action_tile "cancel" "(done_dialog)")
  (start_dialog)
  (unload_dialog dcl_id)
  (vl-file-delete fname)

  (princ v_lst)
)

