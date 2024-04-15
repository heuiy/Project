
(defun save_pdf (filename)
 (setq layout (getvar "CTAB"))
 (command "_.-EXPORT" "_PDF" "_All" "" filename)
 (setvar "CTAB" layout)
)
​
(save_pdf "D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\240129 CAD Reading Automation\\DWG\\in\\240412 저장\\drawing1.pdf")
(save_pdf "D:\\#.Secure Work Folder\\BIG\\Project\\23~24Y\\240129 CAD Reading Automation\\DWG\\in\\240412 저장\\drawing2.pdf")
​​