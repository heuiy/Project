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

dp57_df = data_Rev02_df[data_Rev02_df['제품'] == 'DP57']

dp57_Assay_df = dp57_df.dropna(subset=['Assay'])
DP57_Assay_df = dp57_Assay_df[['제품', 'Site', '배치', 'Assay']]
DP57_Assay = dataiku.Dataset("DP57_Assay")
DP57_Assay.write_with_schema(DP57_Assay_df)

dp57_Chiral_df = dp57_df.dropna(subset=['Chiral'])
DP57_Chiral_df = dp57_Chiral_df[['제품', 'Site', '배치', 'Chiral']]
DP57_Chiral = dataiku.Dataset("DP57_Chiral")
DP57_Chiral.write_with_schema(DP57_Chiral_df)

dp57_Total_Impurity_df = dp57_df.dropna(subset=['Total Impurity'])
DP57_Total_Impurity_df = dp57_df[['제품', 'Site', '배치', 'Total Impurity']]
DP57_Total_Impurity = dataiku.Dataset("DP57_Total_Impurity")
DP57_Total_Impurity.write_with_schema(DP57_Total_Impurity_df)

