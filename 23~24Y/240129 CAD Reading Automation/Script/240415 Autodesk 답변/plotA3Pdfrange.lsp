;Plots layouts by range
; By Alan H Feb 2014
(defun AH:pltlays ( / lay numlay numend)
(SETVAR "PDMODE" 0)

(setvar "fillmode" 1)
(setvar "textfill" 1)

(setq alllays (vla-get-Layouts (vla-get-activedocument (vlax-get-acad-object))))
(setq count (vla-get-count alllays))
(if (not AH:getvalsm)(load "Multi Getvals.lsp"))
(setq vals (AH:getvalsm  (list "Enter plot range" "Enter start tab number" 6 4 "1" "Enter end tab number" 6 4 (RTOS COUNT 2 0))))

(setq numlay (atoi (nth 0 vals)))
(setq numend (atoi (nth 1 vals)))

(setq len (+ (- numend numlay) 1))

(setq dwgname (GETVAR "dwgname"))
(setq lendwg (strlen dwgname))
(setq dwgname (substr dwgname 1 (- lendwg 4)))

(repeat len
(vlax-for lay alllays
(if (= numlay (vla-get-taborder lay))
  (setvar "ctab" (vla-get-name lay))
) ; if
(setq pdfname (strcat (getvar "dwgprefix") "pdf\\" dwgname "-" (getvar "ctab")))

) ; for
(setq lay nil)
(setvar "textfill" 1)
(setvar "fillmode" 1)
    (COMMAND "-PLOT"  "Y"  "" "Plot To PDF"
	       "Iso full bleed A3 (420.00 x 297.00 MM)" "m" "LANDSCAPE"  "N"   "W"  "-6,-6" "807,560" "1=2"  "C"
	       "y" "Acad.ctb" "Y"	"n" "n" "n" pdfName "N" "y"
    )
    
(setq numlay (+ numlay 1))
) ; end repeat
) ; defun

(AH:pltlays)