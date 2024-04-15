(defun c:lotto (/ *error* app bco cir clr col hat hgt ins lst num qty rng spc txt s_cmd s_layer) 

  (defun *error* (m) 
    (if (and (= 'vla-object (type col)) (not (vlax-object-released-p col))) 
      (vlax-release-object col)
    )
    (if s_cmd (setvar "cmdecho" s_cmd))
    (if s_layer (setvar "clayer" s_layer))
    (princ (strcat "\nError: " m))
    (princ)
  )

  (setq s_cmd   (getvar "cmdecho")
        s_layer (getvar "clayer")
  )
  (setvar "cmdecho" 0)
  (if (not (tblsearch "layer" "Defpoints")) 
    (command "layer" "M" "Defpoints" "p" "n" "Defpoints" "")
    (setvar "clayer" "Defpoints")
  )
  (setvar "cmdecho" s_cmd)

  (setq qty 6 ;; Number of Balls
        rng '(1 45) ;; Number Range
        clr ;; Ball Colours
            '((10 (236 014 014) (046 000 000)) ;; Numbers less than 10
              (20 (046 236 014) (000 046 000)) ;; Numbers less than 20
              (30 (236 236 014) (046 046 000)) ;; Numbers less than 30
              (40 (046 014 236) (000 000 046)) ;; Numbers less than 40
              (46 (236 014 236) (046 000 046)) ;; Numbers less than 50
             ) ;;------------------------------------------------------------;;
  )
  (cond 
    ((= 4 (logand 4 (cdr (assoc 70 (tblsearch "LAYER" (getvar 'clayer))))))
     (princ "\nCurrent layer locked.")
    )
    (t
     (setq ins (getvar 'viewctr)
           hgt (getvar 'textsize)
           app (vlax-get-acad-object)
           spc (vlax-get-property (vla-get-activedocument app) (if (= 1 (getvar 'cvport)) 'paperspace 'modelspace))
           col (vla-getinterfaceobject app (strcat "autocad.accmcolor." (substr (getvar 'acadver) 1 2)))
     )
     (while (< (length lst) qty) 
       (if (not (member (setq num (apply 'LM:randrange rng)) lst)) 
         (setq lst (cons num lst))
       )
     )
     (foreach num (vl-sort lst '<) 
       (setq cir (vlax-invoke spc 'addcircle ins (* 1.2 hgt))
             hat (vlax-invoke spc 'addhatch acpredefinedgradient "INVSPHERICAL" :vlax-false acgradientobject)
       )
       (vlax-invoke hat 'appendouterloop (list cir))
       (vla-put-gradientcentered hat :vlax-true)
       (vl-some '(lambda (x) (if (< num (car x)) (setq bco x))) clr)
       (apply 'vla-setrgb (cons col (cadr bco)))
       (vla-put-gradientcolor1 hat col)
       (apply 'vla-setrgb (cons col (caddr bco)))
       (vla-put-gradientcolor2 hat col)
       (vla-delete cir)
       (setq txt (vlax-invoke spc 'addtext (itoa num) ins hgt))
       (vla-put-color txt acwhite)
       (vla-put-alignment txt acalignmentmiddlecenter)
       (vla-put-textalignmentpoint txt (vlax-3D-point ins))
       (setq ins (cons (+ (car ins) (* 3 hgt)) (cdr ins)))
     )
     (vlax-invoke app 
                  'zoomcenter
                  (cons (- (car ins) (* 1.5 (1+ qty) hgt)) (cdr ins))
                  (* (1+ qty) hgt)
     )
     (vlax-release-object col)
    )
  )
  (setvar "clayer" s_layer)
  (princ)
)

(defun LM:rand (/ a c m) 
  (setq m   4294967296.0
        a   1664525.0
        c   1013904223.0
        $xn (rem (+ c (* a (cond ($xn) ((getvar 'date))))) m)
  )
  (/ $xn m)
)

(defun LM:randrange (a b) 
  (fix (+ a (* (LM:rand) (- b a -1))))
)

(vl-load-com)
(princ)