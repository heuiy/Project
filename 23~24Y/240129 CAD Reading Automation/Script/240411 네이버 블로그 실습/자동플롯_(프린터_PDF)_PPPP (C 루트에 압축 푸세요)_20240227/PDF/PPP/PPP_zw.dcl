//출력하기 메뉴 2023.12.28. Kim Hyun Chul (kimhyunchul.co.kr)
ppp_dcl_gstar : dialog {label = " 선택 도면폼 출력하기";

 :row {
       :column {

   :boxed_column { label = "모형탭 출력";
      :radio_button { key = "ra1";  label = " [A3] 인쇄 (명령어 : ppp)"; }
      :radio_button { key = "ra2";  label = " [A3] PDF (명령어 : pdf)"; }
      :radio_button { key = "ra3";  label = " [A4] 인쇄 (명령어 : ppp4)"; }
      :radio_button { key = "ra4";  label = " [A4] PDF (명령어 : pdf4)"; }
   }

   :boxed_column { label = "배치탭 출력";
      :radio_button { key = "ra5";  label = " [A3] 인쇄 (명령어 : mmm)"; }
      :radio_button { key = "ra6";  label = " [A3] PDF (명령어 : pdfm)"; }
      :radio_button { key = "ra7";  label = " [A4] 인쇄 (명령어 : mmm4)"; }
      :radio_button { key = "ra8";  label = " [A4] PDF (명령어 : pdfm4)"; }
   }

   :boxed_column { label = "zw 스마트 배치플롯";
      :radio_button { key = "ra9";  label = " ZW CAD 스마트 배치플롯"; }

   }


		}
	}

  ok_cancel;
}