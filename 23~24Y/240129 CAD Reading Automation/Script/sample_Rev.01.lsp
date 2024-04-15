
(defun save_pdf (filename x y)
 (setq layout (getvar "CTAB"))
 (command "_.-EXPORT" "_PDF" "_All" "" filename)
 (setvar "CTAB" layout)
)
​
(save_pdf "C:\\Users\\LG\\Downloads\\drawing1.pdf" 100 200)
(save_pdf "C:\\Users\\LG\\Downloads\\drawing2.pdf" 300 400)

