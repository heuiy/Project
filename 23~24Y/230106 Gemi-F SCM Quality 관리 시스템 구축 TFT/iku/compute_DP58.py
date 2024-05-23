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

dp58_df = data_Rev02_df[data_Rev02_df['제품'] == 'DP58']

dp58_Assay_df = dp58_df.dropna(subset=['Assay'])
DP58_Assay_df = dp58_Assay_df[['제품', 'Site', '배치', 'Assay']]
DP58_Assay = dataiku.Dataset("DP58_Assay")
DP58_Assay.write_with_schema(DP58_Assay_df)

DP58_Chiral_df = dp58_df.dropna(subset=['Chiral'])
DP58_Chiral_df = dp58_df[['제품', 'Site', '배치', 'Chiral']]
DP58_Chiral = dataiku.Dataset("DP58_Chiral")
DP58_Chiral.write_with_schema(DP58_Chiral_df)
