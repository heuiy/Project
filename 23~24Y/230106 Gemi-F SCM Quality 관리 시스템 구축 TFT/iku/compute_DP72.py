# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
data_Rev02 = dataiku.Dataset("data_Rev02")
data_Rev02_df = data_Rev02.get_dataframe()


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

dp72_df = data_Rev02_df[data_Rev02_df['제품'] == 'DP72']


dp72_Assay_df = dp72_df.dropna(subset=['Assay'])
DP72_Assay_df = dp72_Assay_df[['제품', 'Site', '배치', 'Assay']]
DP72_Assay = dataiku.Dataset("DP72_Assay")
DP72_Assay.write_with_schema(DP72_Assay_df)


dp72_Chiral_df = dp72_df.dropna(subset=['Chiral'])
DP72_Chiral_df = dp72_Chiral_df[['제품', 'Site', '배치', 'Chiral']]
DP72_Chiral = dataiku.Dataset("DP72_Chiral")
DP72_Chiral.write_with_schema(DP72_Chiral_df)


dp72_AUI_df = dp72_df.dropna(subset=['AUI'])
DP72_AUI_df = dp72_AUI_df[['제품', 'Site', '배치', 'AUI']]
DP72_AUI = dataiku.Dataset("DP72_AUI")
DP72_AUI.write_with_schema(DP72_AUI_df)


dp72_Total_Impurity_df = dp72_df.dropna(subset=['Total Impurity'])
DP72_Total_Impurity_df = dp72_Total_Impurity_df[['제품', 'Site', '배치', 'Total Impurity']]
DP72_Total_Impurity = dataiku.Dataset("DP72_Total_Impurity")
DP72_Total_Impurity.write_with_schema(DP72_Total_Impurity_df)


dp72_ROI_df = dp72_df.dropna(subset=['ROI'])
DP72_ROI_df = dp72_ROI_df[['제품', 'Site', '배치', 'ROI']]
DP72_ROI = dataiku.Dataset("DP72_ROI")
DP72_ROI.write_with_schema(DP72_ROI_df)


dp72_Impurity_1_df = dp72_df.dropna(subset=['Impurity-1'])
DP72_Impurity_1_df = dp72_Impurity_1_df[['제품', 'Site', '배치', 'Impurity-1']]
DP72_Impurity_1 = dataiku.Dataset("DP72_Impurity-1")
DP72_Impurity_1.write_with_schema(DP72_Impurity_1_df)

