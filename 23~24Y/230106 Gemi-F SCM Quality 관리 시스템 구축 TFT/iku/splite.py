# 데이터 받아서 한 꺼번에 스플릿 한 거

# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
data = dataiku.Dataset("data_Rev02")
df = data.get_dataframe()

# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

DP26_df = df[df['제품'] == 'DP26']
DP37_df = df[df['제품'] == 'DP37']
DP57_df = df[df['제품'] == 'DP57']
DP58_df = df[df['제품'] == 'DP58']
DP67_df = df[df['제품'] == 'DP67']
DP72_df = df[df['제품'] == 'DP72']

DP26_Assay_df = DP26_df.dropna(subset=['Assay'])
DP26_Assay_df = DP26_Assay_df[['제품', 'Site', '배치', 'Assay']]
DP26_Assay = dataiku.Dataset("DP26_Assay")
DP26_Assay.write_with_schema(DP26_Assay_df)

DP26_Triester_df = DP26_df.dropna(subset=['불순물(Triester)'])
DP26_Triester_df = DP26_Triester_df[['제품', 'Site', '배치', '불순물(Triester)']]
DP26_Triester = dataiku.Dataset("DP26_Triester")
DP26_Triester.write_with_schema(DP26_Triester_df)

DP37_Assay_df = DP37_df.dropna(subset=['Assay'])
DP37_Assay_df = DP37_Assay_df[['제품', 'Site', '배치', 'Assay']]
DP37_Assay = dataiku.Dataset("DP37_Assay")
DP37_Assay.write_with_schema(DP37_Assay_df)

DP57_Assay_df = DP57_df.dropna(subset=['Assay'])
DP57_Assay_df = DP57_Assay_df[['제품', 'Site', '배치', 'Assay']]
DP57_Assay = dataiku.Dataset("DP57_Assay")
DP57_Assay.write_with_schema(DP57_Assay_df)

DP57_Chiral_df = DP57_df.dropna(subset=['Chiral'])
DP57_Chiral_df = DP57_Chiral_df[['제품', 'Site', '배치', 'Chiral']]
DP57_Chiral = dataiku.Dataset("DP57_Chiral")
DP57_Chiral.write_with_schema(DP57_Chiral_df)

DP57_Total_Impurity_df = DP57_df.dropna(subset=['Total Impurity'])
DP57_Total_Impurity_df = DP57_Total_Impurity_df[['제품', 'Site', '배치', 'Total Impurity']]
DP57_Total_Impurity = dataiku.Dataset("DP57_Total_Impurity")
DP57_Total_Impurity.write_with_schema(DP57_Total_Impurity_df)

DP58_Assay_df = DP58_df.dropna(subset=['Assay'])
DP58_Assay_df = DP58_Assay_df[['제품', 'Site', '배치', 'Assay']]
DP58_Assay = dataiku.Dataset("DP58_Assay")
DP58_Assay.write_with_schema(DP58_Assay_df)

DP58_Chiral_df = DP58_df.dropna(subset=['Chiral'])
DP58_Chiral_df = DP58_Chiral_df[['제품', 'Site', '배치', 'Chiral']]
DP58_Chiral = dataiku.Dataset("DP58_Chiral")
DP58_Chiral.write_with_schema(DP58_Chiral_df)

DP67_AUI_df = DP67_df.dropna(subset=['AUI'])
DP67_AUI_df = DP67_AUI_df[['제품', 'Site', '배치', 'AUI']]
DP67_AUI = dataiku.Dataset("DP67_AUI")
DP67_AUI.write_with_schema(DP67_AUI_df)

DP67_Total_Impurity_df = DP67_df.dropna(subset=['Total Impurity'])
DP67_Total_Impurity_df = DP67_Total_Impurity_df[['제품', 'Site', '배치', 'Total Impurity']]
DP67_Total_Impurity = dataiku.Dataset("DP67_Total_Impurity")
DP67_Total_Impurity.write_with_schema(DP67_Total_Impurity_df)

DP72_Assay_df = DP72_df.dropna(subset=['Assay'])
DP72_Assay_df = DP72_Assay_df[['제품', 'Site', '배치', 'Assay']]
DP72_Assay = dataiku.Dataset("DP72_Assay")
DP72_Assay.write_with_schema(DP72_Assay_df)

DP72_Chiral_df = DP72_df.dropna(subset=['Chiral'])
DP72_Chiral_df = DP72_Chiral_df[['제품', 'Site', '배치', 'Chiral']]
DP72_Chiral = dataiku.Dataset("DP72_Chiral")
DP72_Chiral.write_with_schema(DP72_Chiral_df)

DP72_AUI_df = DP72_df.dropna(subset=['AUI'])
DP72_AUI_df = DP72_AUI_df[['제품', 'Site', '배치', 'AUI']]
DP72_AUI = dataiku.Dataset("DP72_AUI")
DP72_AUI.write_with_schema(DP72_AUI_df)

DP72_Total_Impurity_df = DP72_df.dropna(subset=['Total Impurity'])
DP72_Total_Impurity_df = DP72_Total_Impurity_df[['제품', 'Site', '배치', 'Total Impurity']]
DP72_Total_Impurity = dataiku.Dataset("DP72_Total_Impurity")
DP72_Total_Impurity.write_with_schema(DP72_Total_Impurity_df)

DP72_ROI_df = DP72_df.dropna(subset=['ROI'])
DP72_ROI_df = DP72_ROI_df[['제품', 'Site', '배치', 'ROI']]
DP72_ROI = dataiku.Dataset("DP72_ROI")
DP72_ROI.write_with_schema(DP72_ROI_df)

DP72_Impurity_1_df = DP72_df.dropna(subset=['Impurity-1'])
DP72_Impurity_1_df = DP72_Impurity_1_df[['제품', 'Site', '배치', 'Impurity-1']]
DP72_Impurity_1 = dataiku.Dataset("DP72_Impurity-1")
DP72_Impurity_1.write_with_schema(DP72_Impurity_1_df)


