import ROOT as rt
import pandas as pd


df_total  = pd.DataFrame(columns=['Name', 'Value', 'Uncertainty'])
cat_ix =1
# for cat_ix in range(5):
ws = rt.TFile(f"../ucsd_workspace/workspace_bkg_cat{cat_ix}_ggh.root")["w"]
# ws.obj("exp_order2_cat_ggh_coef1").Print("v")
# ws.obj("exp_order2_cat_ggh_coef2").Print("v")
# ws.obj("exp_order2_cat_ggh_frac1").Print("v")
# print(ws.obj("exp_order2_cat_ggh_coef1").getVal())
# print(ws.obj("exp_order2_cat_ggh_coef2").getVal())
# print(ws.obj("exp_order2_cat_ggh_frac1").getVal())
ws.obj("fewz_1j_spl_cat_ggh_pdf_transfer_order2_coef_2_cat1_ggh").Print("v")

raise ValueError
    # if cat_ix ==0:
    #     n_coeff = 3
    # else:
    #     n_coeff = 2
    # for coeff_ix in range(n_coeff):
        # coeff_name = f"fewz_1j_spl_cat_ggh_pdf_transfer_order{n_coeff}_coef_{coeff_ix+1}_cat{cat_ix}_ggh"
        # print(f"coeff_name: {coeff_name}")
        #  ws.obj(coeff_name).Print("v")
        # ws.obj("data_cat0_ggh").Print("v")
    
#         raise ValueError
#         var = ws.obj(coeff_name)
#         # print(f"obj.getVal(): {var.getVal()}")
#         # print(f"obj.getError(): {var.getError()}")
#         df_total = pd.concat([df_total, pd.DataFrame({
#             'Name': [var.GetName()],
#             'Value': [var.getVal()],
#             'Uncertainty': [var.getError()]
#         })], ignore_index=True)

# df_total.to_csv("variables.csv")