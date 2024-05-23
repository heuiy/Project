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

dp26_df = data_Rev02_df[data_Rev02_df['제품'] == 'DP26']

dp26_Assay_df = dp26_df.dropna(subset=['Assay'])
DP26_Assay_df = dp26_Assay_df[['제품', 'Site', '배치', 'Assay']]
DP26_Assay = dataiku.Dataset("DP26_Assay")
DP26_Assay.write_with_schema(DP26_Assay_df)

DP26_Triester_df = dp26_df.dropna(subset=['불순물(Triester)'])
DP26_Triester_df = dp26_df[['제품', 'Site', '배치', '불순물(Triester)']]
DP26_Triester = dataiku.Dataset("DP26_Triester")
DP26_Triester.write_with_schema(DP26_Triester_df)
