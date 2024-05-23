# 과제에는 미반영됨
# 과제에는 각 제품별, 품질 항목 하나씩 추가했음

# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
data_Rev01 = dataiku.Dataset("data_Rev01")
data_Rev01_df = data_Rev01.get_dataframe()

# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

# DP26
# DP37
# DP57
# DP58
# DP67
# DP72

# 제품별로 데이터프레임을 나누고 빈 칼럼 제거하는 함수
def split_and_clean_df(df, product_col):
   product_dfs = {}
   products = df[product_col].unique()

   for product in products:
       product_df = df[df[product_col] == product]
       # 모든 값이 NaN인 칼럼 제거
       product_df = product_df.dropna(axis=1, how='all')
       product_dfs[product] = product_df

   return product_dfs

# '제품' 칼럼을 기준으로 데이터프레임 분리 및 정리
product_dfs = split_and_clean_df(data_Rev01_df, '제품') 
dp26_df = product_dfs.get('DP26')

# Write recipe outputs
DP26 = dataiku.Dataset("DP26")
DP26.write_with_schema(dp26_df)  
