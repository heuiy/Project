//����ϱ� �޴� 2023.12.28. Kim Hyun Chul (kimhyunchul.co.kr)
ppp_dcl_gstar : dialog {label = " ���� ������ ����ϱ�";

 :row {
       :column {

   :boxed_column { label = "������ ���";
      :radio_button { key = "ra1";  label = " [A3] �μ� (��ɾ� : ppp)"; }
      :radio_button { key = "ra2";  label = " [A3] PDF (��ɾ� : pdf)"; }
      :radio_button { key = "ra3";  label = " [A4] �μ� (��ɾ� : ppp4)"; }
      :radio_button { key = "ra4";  label = " [A4] PDF (��ɾ� : pdf4)"; }
   }

   :boxed_column { label = "��ġ�� ���";
      :radio_button { key = "ra5";  label = " [A3] �μ� (��ɾ� : mmm)"; }
      :radio_button { key = "ra6";  label = " [A3] PDF (��ɾ� : pdfm)"; }
      :radio_button { key = "ra7";  label = " [A4] �μ� (��ɾ� : mmm4)"; }
      :radio_button { key = "ra8";  label = " [A4] PDF (��ɾ� : pdfm4)"; }
   }

   :boxed_column { label = "zw ����Ʈ ��ġ�÷�";
      :radio_button { key = "ra9";  label = " ZW CAD ����Ʈ ��ġ�÷�"; }

   }


		}
	}

  ok_cancel;
}