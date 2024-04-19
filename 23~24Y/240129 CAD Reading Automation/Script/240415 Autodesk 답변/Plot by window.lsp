(defun c:inX ()
  (VEREC)
(setq flt '((-4 . "<AND")
	    (0 . "LWPOLYLINE")
	    (8 . "KHUNGTEN")
	    (90 . 4)
	    (70 . 1)
            (-4 . "AND>")
	    );quote
);setq
(if (setq ss (ssget "X" flt));setq
   (progn
      (setq sl (sslength ss))
      (setq s1 0)
      (WHILE (< s1 sl)
	(setq dt2 (SSNAME ss s1))
	(setq A (entget dt2))
	(setq j 14)
	(while (/= (car (nth J A)) 10)
	  (setq j (1+ j))
	  );end while nth
	(setq DINH1 (cdr (nth J A)))
	(setq DINH2 (cdr (nth (+ J 4) A)));
	(setq DINH3 (cdr (nth (+ J 8) A)));
	(setq DINH4 (cdr (nth (+ J 12) A)));
       	(setq GOC1 (ANGLE DINH1 DINH2))
        (setq GOC2 (ANGLE DINH2 DINH3))
        (setq GOC3 (ANGLE DINH3 DINH4))
        (setq GOC4 (ANGLE DINH4 DINH1))
        (setq KC12 (DISTANCE DINH1 DINH2))
        (setq KC14 (DISTANCE DINH1 DINH4))
       (if (= (distance DINH1 DINH3) (distance DINH2 DINH4))
	 (progn 
	   (cond
	     (		;cond1
  	          (and  (= GOC1 0)
			(> KC12 KC14)
			);end AND
		  (Landscape)
	      );end cond1
	     (		;cond2
	          (and  (= GOC1 pi)
			(> KC12 KC14)
			);end and
		  (Landscape)
	      );end cond2
	     (		;cond3
		(and (= GOC1 (/ pi 2))
		     (< KC12 KC14)
		     );end=
		(Landscape)
	      );end cond3
	     (		;cond4
	        (and    (= GOC1 (* pi 1.5))
			(< KC12 KC14)
			);end and
		(Landscape)
	      );end cond4
	     (		;cond5
	        (and   (= GOC1 (/ pi 2))
		       (> KC12 KC14)
		       );end and
		(Portrait)
	      );end cond5
	     (		;cond6
	        (and (= GOC1 (* pi 1.5))
		     (> KC12 KC14)
		     );end=
		(Portrait)
	      );end cond6
	     (		;cond7
	      (and  (= GOC1 0)
		    (< KC12 KC14)
		    );end and
		(Portrait)
	      );end cond7
	   (		;cond8
	    (and (= GOC1 pi)
		 (< KC12 KC14)
		 );end=
   	    (Portrait)
	    );end cond8
	     );end cond
	   );END progn2
	 );END IF2
    (setq s1 (+ s1 1))
    );END WHILE
      (princ "\nPlotting done.")
      );end progn ssget
  (princ "\nNo objects plotting.You need repeat command.")
  );end if
  (princ)
  );end defun
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;--------------------------------------------------------------------
(defun VEREC ()
  
  (setq flt '((-4 . "<AND")
	      (0 . "LWPOLYLINE")
	      (90 . 4)
	      (70 . 1)
	      (-4 . "AND>")
	      );quote
	);setq
  (prompt"\nSelect all drawing change LWPOLYLINE to RECTANG and Plot")
  (if (setq ss (ssget flt));  
    (progn
      (SETQ d1 (sslength ss))
      (SETQ s1 0)
      (command "-layer" "NEW" "KHUNGTEN" "SET" "KHUNGTEN" "C" "123" "" "" "")
        (WHILE (< s1 d1)
	  (progn	;progn1
	    (setq ss2 (ssname ss s1))
	    (setq A (entget ss2))
	    (setq j 14)
	    (while (/= (car (nth J A)) 10)
	      (setq j (1+ j))
	      );end while nth
	    (setq DINH1 (cdr (nth J A)))
	    (setq DINH2 (cdr (nth (+ J 4) A)));
	    (setq DINH3 (cdr (nth (+ J 8) A)));
	    (setq DINH4 (cdr (nth (+ J 12) A)));
	    (COMMAND "RECTANG" DINH1 DINH3)
	    (COMMAND "ERASE" ss2 "")
	    );progn
	(SETQ s1 (1+ s1 ))
	);while
      );progn
    (princ "\nNo objects redraw.You need repeat command.")
    );if
  );end defun
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;---------------------------------------------------------------
(defun Landscape ()
(command "-PLOT"
	 "YES"	;Detailed plot configuration?
	 ""		;Enter a layout name or [?] <Model>:
	 "Canon LBP-4"
	 "A4"
	 "Millimeters"
	 "Landscape"
	 "N"	;Plot upside down?
	 "WINDOW" DINH1 DINH3
	 "FIT"   ;(Plotted Millimeters=Drawing Units) OR [Fit]
	 "CENTER"	;Enter plot offset (x,y) or [Center]
	 "YES"	;Plot with plot styles
	 "MONOCHROME"	;plot style table name
	 "YES"	;PLOT WITH LIGHWEIGHT
	 "As displayed" ;SHADE PLOT SETTING
	 "N"		;Write the plot to a file [Yes/No] <N>:
	 "Y" ;Save changes to page setup [Yes/No]? <N> y
	 "Y" ;Proceed with plot [Yes/No] <Y>:
	 );END command
  )
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;'
;------------------------------------------------------------------
(defun Portrait ()
(command "-PLOT"
	 "YES"	;Detailed plot configuration?
	 ""		;Enter a layout name or [?] <Model>:
	 "Canon LBP-4"
	 "A4"
	 "Millimeters"
	 "Portrait"
	 "N"	;Plot upside down?
	 "WINDOW" DINH1 DINH3
	 "FIT"   ;(Plotted Millimeters=Drawing Units) OR [Fit]
	 "CENTER"	;Enter plot offset (x,y) or [Center]
	 "YES"	;Plot with plot styles
	 "MONOCHROME"	;plot style table name
	 "YES"	;PLOT WITH LIGHWEIGHT
	 "As displayed" ;SHADE PLOT SETTING
	 "N"		;Write the plot to a file [Yes/No] <N>:
	 "Y" ;Save changes to page setup [Yes/No]? <N> y
	 "Y" ;Proceed with plot [Yes/No] <Y>:
	 );END comman
  )
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;-------------------------------------------------------------------