# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
data_Rev02 = dataiku.Dataset("data_Rev02")
data_Rev02_df = data_Rev02.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

dp67_df = data_Rev02_df[data_Rev02_df['제품'] == 'DP67']

dp67_AUI_df = dp67_df.dropna(subset=['AUI'])
DP67_AUI_df = dp67_AUI_df[['제품', 'Site', '배치', 'AUI']]
DP67_AUI = dataiku.Dataset("DP67_AUI")
DP67_AUI.write_with_schema(DP67_AUI_df)

DP67_Total_Impurity_df = dp67_df.dropna(subset=['Total Impurity'])
DP67_Total_Impurity_df = dp67_df[['제품', 'Site', '배치', 'Total Impurity']]
DP67_Total_Impurity = dataiku.Dataset("DP67_Total_Impurity")
DP67_Total_Impurity.write_with_schema(DP67_Total_Impurity_df)

