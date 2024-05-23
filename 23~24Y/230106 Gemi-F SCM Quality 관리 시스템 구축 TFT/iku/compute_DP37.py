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

dp37_df = data_Rev02_df[data_Rev02_df['제품'] == 'DP37']

# Assay 값이 없는 행 제거
dp37_df = dp37_df.dropna(subset=['Assay'])

# 필요한 컬럼만 선택
DP37_Assay_df = dp37_df[['제품', 'Site', '배치', 'Assay']]

# Write recipe outputs
DP37_Assay = dataiku.Dataset("DP37_Assay")
DP37_Assay.write_with_schema(DP37_Assay_df)
