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

dp37_df = data_Rev02_df[data_Rev02_df['제품'] == 'DP37']

dp37_df = dp37_df.dropna(subset=['Assay'])
DP37_Assay_df = dp37_df[['제품', 'Site', '배치', 'Assay']]
DP37_Assay = dataiku.Dataset("DP37_Assay")
DP37_Assay.write_with_schema(DP37_Assay_df)

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

dp58_df = data_Rev02_df[data_Rev02_df['제품'] == 'DP58']

dp58_Assay_df = dp58_df.dropna(subset=['Assay'])
DP58_Assay_df = dp58_Assay_df[['제품', 'Site', '배치', 'Assay']]
DP58_Assay = dataiku.Dataset("DP58_Assay")
DP58_Assay.write_with_schema(DP58_Assay_df)

DP58_Chiral_df = dp58_df.dropna(subset=['Chiral'])
DP58_Chiral_df = dp58_df[['제품', 'Site', '배치', 'Chiral']]
DP58_Chiral = dataiku.Dataset("DP58_Chiral")
DP58_Chiral.write_with_schema(DP58_Chiral_df)

dp67_df = data_Rev02_df[data_Rev02_df['제품'] == 'DP67']

dp67_AUI_df = dp67_df.dropna(subset=['AUI'])
DP67_AUI_df = dp67_AUI_df[['제품', 'Site', '배치', 'AUI']]
DP67_AUI = dataiku.Dataset("DP67_AUI")
DP67_AUI.write_with_schema(DP67_AUI_df)

DP67_Total_Impurity_df = dp67_df.dropna(subset=['Total Impurity'])
DP67_Total_Impurity_df = dp67_df[['제품', 'Site', '배치', 'Total Impurity']]
DP67_Total_Impurity = dataiku.Dataset("DP67_Total_Impurity")
DP67_Total_Impurity.write_with_schema(DP67_Total_Impurity_df)

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

기존 관리도 그리는 함수

st.title('Hello Streamlit!')

# 서브플롯 설정 (24행 3열)
fig, axes = plt.subplots(nrows=24, ncols=3, figsize=(45, 5 * 24))
axes = axes.flatten()

# 이동 평균 계산을 위한 윈도우 크기 설정
window_size = 5

# 넬슨 법칙 탐지 함수들
def check_rule_1(series, mean, std):
   return (series > mean + 3 * std) | (series < mean - 3 * std)

def check_rule_2(series, mean):
   return (series > mean).rolling(window=9).sum().ge(9) | \
          (series < mean).rolling(window=9).sum().ge(9)

def check_rule_3(series):
   return series.diff().gt(0).rolling(window=6).sum().ge(6) | \
          series.diff().lt(0).rolling(window=6).sum().ge(6)

def check_rule_4(series, mean, std):
   two_std = (series > mean + 2 * std) | (series < mean - 2 * std)
   return two_std.rolling(window=14).apply(lambda x: np.sum(x) >= 2, raw=True).fillna(0).astype(bool)

for i, (name, df) in enumerate(dataframes.items()):
   # '익산'을 'Iksan'으로 변경
   name = name.replace('익산', 'Iksan')

   # Individual Chart & ARIMA (왼쪽)
   ax1 = axes[i*3]
   if 'Batch No' in df.columns and '결과값' in df.columns:
       series = df['결과값']
       model = ARIMA(series, order=(1, 1, 1))
       model_fit = model.fit()
       prediction = model_fit.forecast(steps=1)

       mean = series.mean()
       std = series.std()

       # 이동 평균 계산
       moving_average = series.rolling(window=window_size).mean()

       ax1.plot(df['Batch No'], series, marker='o', color='blue')
       ax1.plot(df['Batch No'], moving_average, color='red', linestyle='--')  # 이동 평균 추가
       ax1.scatter(len(df), prediction.iloc[0], color='red')  # ARIMA 예측값 표시
       ax1.axhline(mean, color='green', linestyle='--')
       ax1.axhline(mean + std, color='red', linestyle='--')
       ax1.axhline(mean - std, color='blue', linestyle='--')
       ax1.set_title(f"{name} - Individual & ARIMA", fontsize=10)
       ax1.set_xlabel('Batch No', fontsize=8)
       ax1.set_ylabel('Value', fontsize=8)
       ax1.tick_params(axis='x', labelrotation=90)
       ax1.grid(True)

   # Nelson Rule (가운데)
   ax2 = axes[i*3 + 1]
   if 'Batch No' in df.columns and '결과값' in df.columns:
       anomalies_rule1 = check_rule_1(series, mean, std)
       anomalies_rule2 = check_rule_2(series, mean)
       anomalies_rule3 = check_rule_3(series)
       anomalies_rule4 = check_rule_4(series, mean, std)

       ax2.scatter(df['Batch No'][anomalies_rule1], series[anomalies_rule1], color='red')    # 규칙 1 위반
       ax2.scatter(df['Batch No'][anomalies_rule2], series[anomalies_rule2], color='purple') # 규칙 2 위반
       ax2.scatter(df['Batch No'][anomalies_rule3], series[anomalies_rule3], color='green')  # 규칙 3 위반
       ax2.scatter(df['Batch No'][anomalies_rule4], series[anomalies_rule4], color='orange') # 규칙 4 위반
       ax2.axhline(mean, color='green', linestyle='--')
       ax2.axhline(mean + std, color='red', linestyle='--')
       ax2.axhline(mean - std, color='blue', linestyle='--')
       ax2.set_title(f"{name} - Nelson Rules", fontsize=10)
       ax2.set_xlabel('Batch No', fontsize=8)
       ax2.set_ylabel('Value', fontsize=8)
       ax2.tick_params(axis='x', labelrotation=90)
       ax2.grid(True)

   # Moving Average (오른쪽)
   ax3 = axes[i*3 + 2]
   if 'Batch No' in df.columns and '결과값' in df.columns:
       moving_range = df['결과값'].diff().abs()
       moving_avg = moving_range.rolling(window=window_size).mean()

       ax3.plot(df['Batch No'][1:], moving_range[1:], marker='o', color='blue', alpha=0.5)  # 원본 데이터
       ax3.plot(df['Batch No'][window_size:], moving_avg[window_size:], color='red')  # 이동 평균
       ax3.set_title(f"{name} - Moving Average", fontsize=10)
       ax3.set_xlabel('Batch No', fontsize=8)
       ax3.set_ylabel('Moving Range', fontsize=8)
       ax3.tick_params(axis='x', labelrotation=90)
       ax3.grid(True)

plt.tight_layout()
plt.show()
