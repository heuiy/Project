# ___________________________--------
# **************************************--------
# ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁====
# 03 _ 210319 MBB 실습 과제 ----
# ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁====
# ___________________________--------  
# **************************************--------

# BPA----

# 실습과제 ------------------------------------------------------------
# 작성자 :      
# Text encoding : UTF-8

# 세부 환경 설정 ---------------------------------------------------------------

dir()
# [1] "BPA.R"   "BPA.Rproj"  "Bucket_MBB.csv"  "LIMS_MBB.csv"   "pis_raw_MBB.csv"

# set Working directory 
# getwd()
# setwd("D:/#.Secure Work Folder/DX-LSS-Project/DX-LSS MBB simulation/BPA")
# R Studio에서 Session > Set working directory > choose directory 로 설정 가능


# ■ Define ---------------------------------------------------------------------

# Step01 개선 기회 탐색  -------------------------------------------------------
# - LIMS 시스템에서 과제 도출 가능?


# Step02 개선 기회 발굴 및 과제 선정--------------------------------------------
# - Big Y - little y 전개를 통해 과제 선정?


# Step03 Project Y 선정  -------------------------------------------------------
## KPI
## CTQ


# ■ Measure ----------------------------------------------------------------------
# Step04 데이터 수집 및 검증 계획 수립   ---------------------------------------
# 

# Step05 데이터 Set 구성 -------------------------------------------------------

# 전처리 도구 불러오기
library(dplyr);library(tidyr)

# 데이터 불러오기 
lims=read.csv("LIMS_MBB.csv")   # 내장 함수

# library(readr)
# lims=read_csv("LIMS_MBB.csv")   
   



# 데이터 체크
head(lims)
str(lims)

# lims 데이터 중 필요한 데이터만 선택
colnames(lims)

lims1 = lims[,c(1,22,24,25)]
#lims1 = lims %>%  select(`Date/Time`, `PSD (LT 0.15mm)`, `PSD (LT 0.85mm)`, `PSD (0.85~2.0mm)`)

# lims 데이터 중 project Y만 선택 Date.Time, PSD..LT.0.15mm., PSD..LT.0.85mm., PSD..0.85.2.0mm.
head(lims1) # 변수 선택 저장 결과 체크


# D50 계산식 추가
str(lims1) # PSD.. 0.85. 2. 0mm가 Factor로 들어가있으니, numeric으로 변경



lims1[,4]
lims1[,4] = as.numeric(as.character(lims[,4])) 
lims1$PSD..0.85.2.0mm. = as.numeric(as.character(lims$PSD..0.85.2.0mm.))  

# Factor는 반드시 character로 변환 후 numeric으로 변환시켜야 함.
# (아니면 다른 값이 됨)

str(lims1) #변환 결과 체크 


# D50을 구하는 공식은 식은 내부에서 공식 활용
# D50 = 850 + ( 50 - 'PSD(LT 0.85mm)' ) / 'PSD(0.85~2.00)' X (2000-850)

lims1=lims1 %>% mutate(D50=(850+(50-PSD..LT.0.85mm.)/PSD..0.85.2.0mm.*(2000-850))) 
head(lims1) # 변환 결과 확인
#tail(lims1)
#dim(lims1)

# Step07 프로세스 현수준 파악  -------------------------------------------------
head(lims1)
lims1=lims1[,c(1,2,5)] # 사용할 변수만 저장. select 명령을 사용해도 됨 
#colnames(lims1)
#lims1 <- lims1 %>% select(Date.Time,PSD..LT.0.15mm.,D50)
#head(lims1)
#summary(lims1)

library(SixSigma)
#ss.study.ca(xST, xLT = NA, LSL = NA, USL = NA, Target = NA)
lims1 <- na.omit(lims1)
ss.study.ca(xST=lims1$D50 , LSL = 1300, Target = 1600  )



# Step08 개선 목표 설정  -------------------------------------------------------
# D50 평균 1434  ->  목표 평균 1600 


# ■  Analyze ----------------------------------------------------------------------
# Step09 X인자 검증 계획 수립  -------------------------------------------------
# 데이터 수집 계획


# Step10 데이터 취득 및 전처리 실시  -------------------------------------------

# 전처리 도구 불러오기
library(dplyr);library(tidyr)

bucket=read_csv("Bucket_MBB.csv")
pis_product = read_csv("pis_raw_MBB.csv")

head(bucket)
head(pis_product)
str(pis_product)

# saveRDS(bucket, file = 'bucket')  # bucket 데이터 셋 저장
# saveRDS(pis_product, file = 'pis_product')  # pis_product 데이터 셋 저장

# rm(list=ls())  #데이터셋 모두 삭제
# dir()
# lims <- readRDS('lims') # 데이터 셋 불러오기 
# bucket <- readRDS('bucket') # 데이터 셋 불러오기
# pis_product <- readRDS('pis_product') # 데이터 셋 불러오기


#날짜 데이터 변환하기 
str(pis_product$Date.Time)
pis_product$Date.Time = as.POSIXct(pis_product$Date.Time)
str(pis_product)

str(lims1)
lims1$Date.Time
lims1$Date.Time = as.POSIXct(lims1$Date.Time)
str(lims1)


# 날짜 형식 전처리 
head(bucket)
colnames(bucket)[1]
colnames(bucket)[1]=c("Date.Time")

bucket$Date.Time
bucket$Date.Time = as.POSIXct(bucket$Date.Time)
str(bucket)


#format을 사용해 날짜형식을 변환시킬때는 POSIXct 형식으로 변환되어 있어야 오류가 없음
pis_product[,'Date.Time']=pis_product$Date.Time %>% format(.,"%Y-%m-%d")
# pis_product$Date.Time <- pis_product$Date.Time %>% format(.,"%Y-%m-%d")

# pis_product[,1]=pis_product[,1] %>% format(.,"%Y-%m-%d") 
lims1[,'Date.Time']=lims1$Date.Time %>% format(.,"%Y-%m-%d") 
bucket[,'Date.Time']=bucket$Date.Time %>% format(.,"%Y-%m-%d") 

str(pis_product)
str(lims1)
str(bucket)
#


#데이터 병합(merge방식)
bpa_dataset1 = merge(lims1, pis_product, by='Date.Time')  #lims와 pis를 먼저 
head(bpa_dataset1,n=1)

bpa_dataset2 = merge(bpa_dataset1,bucket,by='Date.Time')
dim(bpa_dataset2);dim(lims);dim(bucket);dim(pis_product)

sum(is.na(bpa_dataset2))

df = na.omit(bpa_dataset2)


# Step06 데이터 취득 시스템(유용성)검증  ---------------------------------------

#데이터셋(이상치 제거전) 변수, 데이터 수 체크 
dim(df)


#y값(D50) 이상치 제거하기
colnames(df)
names(df)[2]
names(df)[2]=c("LT0.15")


boxplot(df$D50)
# abline(h=c(1224.411,1659.229),col="red",lty="dotted") 

boxplot(df$D50)$stats
# boxplot.stats(df$D50) 두가지 차이를 실행시켜서 확인해보세요. 

boxplot(df$D50)$stats[1] # Q1-IQR*1.5 
# 이상치 계산 : quantile(df$D50, 1/4) - IQR(df$D50)*1.5
boxplot(df$D50)$stats[5] # Q3+IQR*1.5
# 이상치 계산 : quantile(df$D50, 3/4) + IQR(df$D50)*1.5


df = df %>% filter(D50>boxplot(D50)$stats[1],D50<boxplot(D50)$stats[5]) %>% select(.,-Date.Time)

# df = df %>% filter(D50>1223.741,D50<1659.229) %>% select(.,-Date.Time) 

summary(df)
colnames(df)
dim(df)



# 최종 Dataset 저장 
#saveRDS(df, file = 'df')  # 최종 데이터 set 저장
#rm(list=ls()) # 모든 데이터 셋 삭제
df <- readRDS('df') # df 불러오기 


# Step11 데이터 탐색  ----------------------------------------------------------

# 데이터 요약

#graph분석
pairs(df)

# 다른 그래프 
df_cor <- cor(df)
df_cor
library(corrplot)
corrplot(df_cor)  

# 또다른 그래프 
panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}


pairs(df %>% sample_n(min(1000, nrow(df))),
      lower.panel=function(x,y){ points(x,y); abline(0, 1, col='red')},
      upper.panel = panel.cor)



# Step12 핵심인자 선정  --------------------------------------------------------
## 데이터 Set 구분하기

#Data split

dim(df)
nrow(df)

set.seed(0319) # 샘플링할때 약속된 데이터를 동일하게 샘플링하도록 
sample(1:5,3)
nrow(df)*0.7
train=sample(nrow(df),nrow(df)*0.7)
train

test=(1:nrow(df))[-train]
test

length(train)
length(test)


# Step13 분석 모형 검토  -------------------------------------------------------

# 아래는 각 인자별 관계를 확인하지 않고 그냥 DO가 Project Y 나머지를 그냥 기존 데이터로 생각하고 돌린 부분입니다. 이와 비슷하게 정리하려고 하오니, 해석에 오해가없도록 참조만 하세요. 

#분석실시(modeling) regression, rf

colnames(df)
df2 <- df[,-1]
colnames(df2)

lm.fit=lm(D50~.,data=df[train,-1])
lm.fit=lm(D50~.,data=df2[train,])
summary(lm.fit)

step(lm.fit)

#최종 회귀식 m(formula = D50 ~ Feed_flow + Feed_Temp + Bucket.RPM_1
        #               + Bucket.RPM_2 + out_AirTemp + Prill.Tower_temp
        #               + Bucke.Hole.Size, data = bpa_set[train,-1])


colnames(df)
lm.fit_best=lm(D50 ~ Feed_flow + Feed_Temp + Bucket.RPM_1 + Bucket.RPM_2 + 
       + out_AirTemp + Prill.Tower_temp + Bucke.Hole.Size, data = df[train, -1])


#실제는 다중 공선성 체크하고, 변수 조정해줘야 함

summary(lm.fit_best) # Multiple R-sq 값 체크,  Estimate(부호 체크), *표시 체크 




library(randomForest) ; library(rpart)
rf.fit=randomForest(D50~.,data=df[train,-1],importance=T)
rf.fit
# importance(rf.fit)
varImpPlot(rf.fit)



# ■ Improve ----------------------------------------------------------------------
# Step14 최적 모형 수립  -------------------------------------------------------


# 상관관계 분석 후 제곱해서 계산, 
# but 실질적으로는 회귀 R-sq와, 랜덤포레스트  %var explained를 보고 판단하는 것이맞음

lm_obs = df[test,]$D50 #실제 관측값
lm_pred = predict(lm.fit_best,newdata=df[test,-1]) # 예측값 
rf_pred = predict(rf.fit,newdata=df[test,-1]) # 예측값 

#(cor(lm_pred,lm_obs))^2  #상관계수 제곱
#(cor(rf_pred,lm_obs))^2  #상관계수 제곱


library(DescTools)
MSE(lm_pred, lm_obs)
MSE(rf_pred, lm_obs)

RMSE(lm_pred, lm_obs) # 회귀식이 상대적으로 우수함
RMSE(rf_pred, lm_obs)

# library(Metrics) # 교육생 문의(6/25)
# bias(lm_pred, lm_obs)

# Step15 모형 검증 및 최적화  --------------------------------------------------

# 모형 선택은 다중선형회귀를 선택함
#최적조건 구하기- 인자별 Range확인 (이상치 제거,  0이하 제거 )
colnames(df)
df_range <- df %>% select(-LT0.15)

print("Feature range")
for( i in 1:ncol(df_range)){
  A = df_range %>% filter(df_range[,i]>0) %>% .[,i] 
  B = A[A>boxplot(A)$stats[1]&A<boxplot(A)$stats[5]] %>% range()
  print(data.frame(names=colnames(df_range)[i] ,lower=B[1],upper=B[2]))
}

df$Bucke.Hole.Size %>% table()  #Buckek.hole.size는 정수로, 범위 별도 체크 



summary(lm.fit_best) # 부호 다시 체크 

#08. 최적 모형 예측값 ----------------------------------------------------------

# 예측 값을 찾기 위해 회귀식 부호 방향을 체크하고, 그 값을 반영함
# 단, 실무적으로 각 변수의 설정값을 검토 하여 조정 가능함(비용, 안정성 측면 고려)

new=data.frame(Feed_flow=15.2507,         # 음
               Feed_Temp=163.8268,        # 음
               Bucket.RPM_1=83.1646,      # 음
               Bucket.RPM_2=93.2791,      # 음
               out_AirTemp=32.5472,       # 양
               Prill.Tower_temp=56.587,   # 양
               Bucke.Hole.Size=1.3,       # 양
               outAir.flow=806.5549)      # 양 (randomForest를 예측값을 위해 넣음)

new


#최적조건 구하기- 인자별 Range확인 (예측) 

predict(lm.fit_best,newdata=new)

# predict(rf.fit,newdata=new)



# Step16 개선 결과 검증(Pilot Test) --------------------------------------------


# ■ Control ----------------------------------------------------------------------
# Step17 최적모형 모니터링  ----------------------------------------------------

# Step18 표준화 및 수평전개  ---------------------------------------------------


#참고) 다중공선성 고려했을때 ------------------

colnames(df)
lm.fit=lm(D50~.,data=df[train,-1])
summary(lm.fit)

step(lm.fit)
lm.fit = lm(D50 ~ Feed_flow + Feed_Temp + Bucket.RPM_1 + Bucket.RPM_2 + 
              out_AirTemp + Prill.Tower_temp + Bucke.Hole.Size, data = df[train, -1])

vif(lm.fit)
#최종 회귀식 (formula = Feed_flow + Feed_Temp + Bucket.RPM_1 + outAir.flow, data = df[train, -1])

#Bucket.RPM 1, 2중 하나만 반영하는 것으로 검토하여 결정 
colnames(df)
lm.fit_best=lm(D50 ~ Feed_flow + Feed_Temp + Bucket.RPM_1 + 
                 out_AirTemp + Prill.Tower_temp + Bucke.Hole.Size, data = df[train, -1])
step(lm.fit_best)
lm.fit_best=lm(D50 ~ Feed_flow + Feed_Temp + Bucket.RPM_1 + out_AirTemp + 
                 Prill.Tower_temp + Bucke.Hole.Size, data = df[train, -1])

summary(lm.fit_best) # Multiple R-sq 값 체크,  Estimate(부호 체크), *표시 체크 


library(randomForest) ; library(rpart)
rf.fit=randomForest(D50~Feed_flow + Feed_Temp + Bucket.RPM_1 + outAir.flow,data=df[train,-1],importance=T)
rf.fit
importance(rf.fit)
varImpPlot(rf.fit)



# 최적 모형 선정
lm_obs = df[test,]$D50 #실제 관측값
lm_pred = predict(lm.fit_best,newdata=df[test,-1]) # 예측값 
rf_pred = predict(rf.fit,newdata=df[test,-1]) # 예측값 

(cor(lm_pred,lm_obs))^2  #상관계수 제곱
(cor(rf_pred,lm_obs))^2  #상관계수 제곱


library(DescTools)
MSE(lm_pred, lm_obs)
MSE(rf_pred, lm_obs)

RMSE(lm_pred, lm_obs) # 회귀식이 상대적으로 우수함
RMSE(rf_pred, lm_obs)


# 모형 최종 예측 값 확인 
colnames(df)
df_range <- df %>% select(-LT0.15, -Bucke.Hole.Size)

print("Feature range")
for( i in 1:nrow(df_range)){
  A = df_range %>% filter(df_range[,i]>0) %>% .[,i] 
  B = A[A>boxplot(A)$stats[1]&A<boxplot(A)$stats[5]] %>% range()
  print(data.frame(names=colnames(df_range)[i] ,lower=B[1],upper=B[2]))
}

df$Bucke.Hole.Size %>% table()  #Buckek.hole.size는 정수로, 범위 별도 체크 

summary(lm.fit_best) # 부호 다시 체크 


colnames(df)
new=data.frame(Feed_flow=15.2507,         # 음
               Feed_Temp=163.8268,        # 음
               Bucket.RPM_1 = 83.16,       # 음
               out_AirTemp = 32.5472,     # 양
               Prill.Tower_temp=56.587,   # 양
               Bucke.Hole.Size=1.3)       # 양



predict(lm.fit_best,newdata=new)

#_______________________----
# overlap----

# 작성자 :      
# Text encoding : UTF-8

dir()

# 아래와 같이 데이터가 뜨면 별도의 working directory를 설정하지 않으셔도 됩니다. 
# [1] "200421_overlap_dataset.xlsx" "200421_overlap_modelA.csv"   "200421_overlap_modelB.csv"  
# [4] "200421_overlap_modelC.csv"   "overlap.R"                   "overlap.Rproj"   



# set Working directory 
getwd()
setwd(" ")
# R Studio에서 Session > Set working directory > choose directory 로 설정 가능



# 02. 데이터 전처리 ------------------------------------------------------------

library(tidyr);library(dplyr)

# 데이터 불러오기 
# Data load
A=read.csv("200421_overlap_modelA.csv")
B=read.csv("200421_overlap_modelB.csv")
D=read.csv("200421_overlap_modelC.csv")


#Data pre-processing1 (평균)
A=(na.omit(A));B=na.omit(B);C=na.omit(C)  #model A데이터 빈칸을 NA로 인식관련 [,1:43]범위 설정 
dim(A);dim(B);dim(D)




#데이터 특성 변환
str(B)
as.character(B$ 용접전압..1.) #용접전압..1.만 Factor로 되어 있어 변환 필요 
B$ 용접전압..1.=as.numeric(B$ 용접전압..1.)
str(B)


A=data.frame(y=A[,3], Model='A',A[1:2],
             overlap=apply(A[4:13],1,mean),
             voltage=apply(A[14:23],1,mean),
             current=apply(A[24:33],1,mean),
             resistor=apply(A[34:43],1,mean))

B=data.frame(y=B[,3], Model='B',B[1:2],
             overlap=apply(B[4:13],1,mean),
             voltage=apply(B[14:23],1,mean),
             current=apply(B[24:33],1,mean),
             resistor=apply(B[34:43],1,mean))

D=data.frame(y=D[,3], Model='C',D[1:2],
             overlap=apply(D[4:13],1,mean),
             voltage=apply(D[14:23],1,mean),
             current=apply(D[24:33],1,mean),
             resistor=apply(D[34:43],1,mean))

head(A);head(B);head(D)
str(A);str(B);str(D)


A$Model=factor(A$Model,levels=c('A','B','C'))  #A, B, C 레벨 설정
A$Model

B$Model=factor(B$Model,levels=c('A','B','C'))
B$Model

D$Model=factor(D$Model,levels=c('A','B','C'))
D$Model


#Data pre-processing3 (delete outlier )  

boxplot(A$y)
boxplot.stats(A$y)$stats[1]
boxplot.stats(A$y)$stats[5]

A=A %>% filter(y<boxplot.stats(A$y)$stats[5]) %>% filter(y>boxplot.stats(A$y)$stats[1])

boxplot(B$y)
boxplot.stats(B$y)
B=B %>% filter(y<boxplot.stats(B$y)$stats[5]) %>% filter(y>boxplot.stats(B$y)$stats[1])

boxplot(D$y)
boxplot.stats(D$y)
D=D %>% filter(y<boxplot.stats(D$y)$stats[5]) %>% filter(y>boxplot.stats(D$y)$stats[1])


#Data pre-processing4 (merge of A,B,C model )
ABD_total=rbind(A,B,D)
ABD=ABD_total %>% select(y,Model,overlap, voltage, current, resistor) #필요 변수 선택
str(ABD)

ABD$Model


# 03. 데이터 탐색 -------------------------------------------------------------
pairs(ABD)


# 04. 데이터 Set 구분하기 ------------------------------------------------------

#data_split (3 vs 7)
set.seed(7279)
train =sample(nrow(ABD), nrow(ABD)*0.7)
test= (1:nrow(ABD))[-train]



#05. 모형 선정 -----------------------------------------------------------------

#modeling-1 (linear model)
ABD %>% head(2)

lm_y=lm(y~.,data=ABD[train,])
lm_y
step(lm_y)

lm_y_best=lm(y ~ Model + overlap + current, data = ABD[train,])

#실제 다중공선성 체크 및 변수에 대한 실질적 조정이 필요함(실제는 voltage로 조정함)



#modeling-2 (randomForest)

library(randomForest)
library(tree)

rf_y=randomForest(y~.,data=ABD[train,],importance=T)
rf_y
importance(rf_y)
varImpPlot(rf_y)

# Tree그림 그리기
tree_y=tree(y~.,data=ABD[train,])
plot(tree_y);text(tree_y)

library(rpart)
tree_y2 = rpart(y~., data=ABD[train,])

par(mar=c(1,1,1,1), xpd = TRUE) 
plot(tree_y2) ; text(tree_y2, use.n = T)


# 06. 모형 선택 --------------------------------------------------------------
lm_obs = ABD$y[test]
lm_pred = predict(lm_y_best, newdata=ABD[test,])
rf_pred = predict(rf_y, newdata=ABD[test,])

(cor(lm_pred,lm_obs))^2  #상관계수 제곱
(cor(rf_pred,lm_obs))^2  #상관계수 제곱


library(DescTools)
MSE(lm_pred, lm_obs)
MSE(rf_pred, lm_obs)

RMSE(lm_pred, lm_obs) 
RMSE(rf_pred, lm_obs)



# 07. 모형 최적화 --------------------------------------------------------------

# 랜덤포레스트가 우수하나 큰 차이가 없어 lm으로 선정
# 최적조건 구하기- 인자별 Range확인

head(ABD,1)

print("Feature range")
for( i in 3:6){
  A = ABD %>% filter(ABD[,i]>0) %>% .[,i] 
  B = A[A>boxplot(A)$stats[1]&A<boxplot(A)$stats[5]] %>% range()
  print(data.frame(names=colnames(ABD)[i] ,lower=B[1],upper=B[2]))
}


summary(lm_y_best)



#08. 최적 모형 예측값 ----------------------------------------------------------

#최적조건 구하기- 인자별 Range확인(예측), 
# 단, 실무적으로 각 변수의 설정값을 검토 하여 조정 가능함(비용, 안정성 측면 고려)

new=data.frame(Model=factor('A',levels=c('A','B','C')),  #A모델(개선대상 A모델임)
               overlap= 2.47 ,    #양
               voltage= 3.15  ,   #양
               current= 1347.5 ,  #양   영향성은 낮지만, 부호방향으로 값 설정
               resistor= 23.2)    #양   영향성은 낮지만, RF관련 Tree기준으로 입력 


predict(lm_y_best,newdata=new)
predict(rf_y,newdata=new)


#_______________________----
# POE----

# 과제 1 DX-LSS Simulation 실습과제 --------------------------------------------
# 작성자 : 백인엽 책임,  수정일 21. 4. 1     
# Text encoding : UTF-8

# 세부 환경 설정 ---------------------------------------------------------------

dir()
# 아래 파일이 보이면 별도 Working directory를 재설정 하지 않으셔도 됩니다. 
# [1] "df0"                   "MBB_LIMS_LINE1.csv"    "MBB_LIMS_LINE2.csv"   
# [4] "MBB_POE_PIS_30000.csv" "POE.R"                 "POE.Rproj"            
# [7] "TAG_INFO.csv"          "TVOC_M.csv"     


# # set Working directory 
# getwd()
setwd("D:/#.Secure Work Folder/DX-LSS-Project/DX-LSS MBB simulation/POE")
# R Studio에서 Session > Set working directory > choose directory 로 설정 가능


# ■ Define ---------------------------------------------------------------------

# Step01 개선 기회 탐색  -------------------------------------------------------
# - LIMS 시스템에서 과제 도출 가능

# Step02 개선 기회 발굴 및 과제 선정--------------------------------------------
# - Big Y - little y 전개를 통해 과제 선정

# Step03 Project Y 선정  -------------------------------------------------------
## KPI  TVOC 300ppm --> 200ppm,   KPI와 CTQ 동일
## Total Volatile Organic Compound
## 자동차 내외장재, 차에서 나는 냄새를 관리, TVOC라는 냄새를 유발하는 휘발성 유기 화합물을 CTQ로 하는 개선 과제임 공정 특성상 노말헥산을 사용하고 있어 솔벤트에 있어 잔류하는 휘발성 액체가 남아 있어 공중합체로 투입되는 옥텐으로 있어  잔류된 노말헥산, 옥텐을 제거 하는 과제임
## TVOC는 제품 Lot당 1개씩 측정됨. 공정 데이터와 비교 데이터가 작아 대용지표 발굴 필요 



# ■ Measure ----------------------------------------------------------------------
# Step04 데이터 수집 및 검증 계획 수립   ---------------------------------------
# 고객社 분석법(VDA277)과 동일한 조건으로 TVOC Data 수집 계획을 수립함
# 측정 지표 : TVOC(ppm), 측정 설비 : GC 측정 설비, 수집 기간 : '20. 1~3월(1분기)



# Step05 데이터 Set 구성 -------------------------------------------------------
z = read.csv("TVOC_M.CSV")
colnames(z)
colnames(z)[3]=c("y")  # 3번열의 이름을 y로 변경함. 


# Step06 데이터 취득 시스템(유용성)검증  ---------------------------------------
# Step07 프로세스 현수준 파악  -------------------------------------------------
#Install.packages(SixSigma)
library(SixSigma)
#ss.study.ca(xST, xLT = NA, LSL = NA, USL = NA, Target = NA)
ss.study.ca(xST=z$y, USL = 300, Target = 200)

# rm(list=ls())
# dev.off()



## project Y인 TVOC 분석 주기에 따른 공정 응답성 지연을 고려하여 VM을 대용지표로 선정함
## cor(z$y, lims$vm) ; pairs(z$y, lims$vm)
## TVOC vs VM에 대한 pearson 상관계수 0.618(p-value 0.000)임 



## VM 선정 배경
### 1. 분석 시간이 짧아 피드백이 빠르며, 공정 제어 지표로서 적절함
### 2. 생산 제품의 현재 휘발분 함량 상태를 즉각적으로 대변함
### 3. 동일주기 측정으로 (매일 07:00) 공정 상태와의 Data 연계가 보다 용이하여 예측 모델의 신뢰성 확보 가능



# 대용 지표 인 VM의 현수준 파악
# 측정 지표 : VM(ppm), 측정 설비 : 약식 분석, 1일 1회(07:00), 수집 기간 : '20. 1~3월(1분기), 측정 위치 : Hold-up bin, 분석 원리 : 가열 전후 Weight감소량(질량 분율), 기준 온도 120도, 총분석 시간 : 2hr

# 데이터 불러오기
library(readr)
lims1 <- read_csv("MBB_LIMS_LINE1.csv")
lims2 <- read_csv("MBB_LIMS_LINE2.csv")
# 측정 지표 : VM(ppm), 측정 설비 : 약식 분석, 1일 1회(07:00), 수집 기간 : '20. 1~3월(1분기), 측정 위치 : Hold-up bin, 분석 원리 : 가열 전후 Weight감소량(질량 분율), 기준 온도 120도, 총분석 시간 : 2hr
str(lims1)

head(lims1,1)
head(lims2,1)

# 데이터 합치기 
library(dplyr)
# lims_all <-  rbind(lims1,lims2)
lims_all <- bind_rows(lims1, lims2)   #rbind와 유사한 함수 
head(lims_all)

lims_all <- lims1  #실제로는 line1에 대해서만 분석 



# 데이터 추출(필요한 변수만 추출)
df1 <- lims_all %>%
  select(TIME, GRADE, MI, VM) 

# TIME변수는 Key로 활용, VM은 Target값임, GRADE, MI는 층별 분석에 활용


# 데이터 속성 확인 및 변환 
glimpse(df1)
df1 <- df1 %>% 
  mutate(MI = as.numeric(MI),   #수치형으로 변환하는 함수
         VM = as.numeric(VM))   #수치형으로 변환하는 함수

# df1$MI <- as.numeric(MI) #단일 변수의 데이터열의 변환할때 사용 


# Target변수 통계적 이상치 NA처리
df1$VM <- ifelse(df1$VM >= 0, df1$VM, NA) # 결측치 NA변환
df1 %>% 
  filter(!is.na(VM)) %>%
  select(VM) %>% 
  boxplot()

boxplot(df1$VM)
boxplot(df1$VM)$stats

df1$VM <- ifelse(df1$VM >= 0 & df1$VM <= boxplot(df1$VM)$stats[5] , df1$VM, NA)

# df1$VM <- ifelse(df1$VM >= 0, df1$VM, NA)  # 1.5*IQR+Q3 값이 이상치가 아니라고 판단될 경우 사용 
  


# 결측치 제외한 Target변수 최종 데이터 크기 확인 

df1 %>% 
  filter(!is.na(VM)) %>%    # na가 아닌 것만 필터링
  count(VM) %>%             # VM의 데이터의 수
  summarise(n = sum(n))     # n의 합계


# VM에 대한 hist그램과 plot확인 
hist(df1$VM)            #히스토그램
plot(df1$TIME, df1$VM)  #VM의 시계열 산점도

df1 <- na.omit(df1)

library(SixSigma)
#ss.study.ca(xST, xLT = NA, LSL = NA, USL = NA, Target = NA)
ss.study.ca(xST=df1$VM, USL = 4000, Target = 3000)


# Step08 개선 목표 설정  -------------------------------------------------------
## VM 평균 3507ppm(Z bench 0.40) -> 평균 3000ppm 이하(Z bench 3.00)


# ■  Analyze ----------------------------------------------------------------------
# Step09 X인자 검증 계획 수립  -------------------------------------------------

# 데이터 수집 계획
# project Y : VM data - LIMS (1회/1일, 07:00)
# x's : 각 TAG data - PIS 공정 Data
# 데이터 Merge 필요 
# library(readr)
# dir()
# TAG = read_csv("TAG_INFO.csv")
# head(TAG)

# 데이터 불러오기 


# 1.2 설비/공정 변수 데이터 전처리
# 데이터 불러오기 
library(readr)
# pis1 <-  read_csv("MBB_POE_PIS.csv") # 원데이터는 Line1 324MB 
# pis2 <-  read_csv("MBB_POE_PIS_LINE2.csv")  # 원데이터는 Line1 403MB 
# df3M <- sample(nrow(pis1), 30000)
# df <- pis1[df3M,]
# write.csv(df, file = "MBB_POE_PIS_30000.csv")
# df$TIME <- as.POSIXct(df$TIME)
# df <- arrange(df,df$TIME)
pis1 <-  read_csv("MBB_POE_PIS_30000.csv")  

colnames(pis1)

df2 <- pis1    # 복사본 만들기 
dim(df2)
str(df2)

# 표준 시간 변환 및 분석 가능 시간으로 변환
library(lubridate)
str(df2)
df2$TIME <- ymd_hm(df2$TIME)                 # 년월일_시분 형식 변환
# 주의사항 POSIXct형태로 변환된 상태에서 ymd_hm을 실행하면 NA처리됨 

df2$TIME <- with_tz(df2$TIME, "Asia/Seoul")  # 표준시간 변경 
df2$TIME <- round_date(df2$TIME, "30 mins")  # 30분 단위 반올림
# round_date(df2$TIME, "30 mins")



?round_date  #round_date의 기능과 옵션을 확인해보세요. 
df3 <- as.POSIXct("2021-04-01 9:14") 
round_date(df3, "30 mins")
# 위의 함수를 실행시켜보고, 
# 9시 14분,15분,44분,45분을 바꿔 넣고 어떻게 반올림 되는지 확인해보세요. 



df2 <-  df2 %>% 
  group_by(TIME) %>%        #시간 기준으로 묶음 
  summarise(P1 = mean(P1),  #시간단위 묶은 데이인 P1의 평균 값 
            P2 = mean(P2),
            P3 = mean(P3),
            P4 = mean(P4),
            P5 = mean(P5),
            P6 = mean(P6),
            P7 = mean(P7),
            P8 = mean(P8),
            P9 = mean(P9),
            P10 = mean(P10),
            P11 = mean(P11),
            P12 = mean(P12),
            P13 = mean(P13),
            P14 = mean(P14),
            P15 = mean(P15),
            P16 = mean(P16),
            P17 = mean(P17),
            P18 = mean(P18),
            P19 = mean(P19),
            P20 = mean(P20),
            P21 = mean(P21),
            P22 = mean(P22),
            P23 = mean(P23),
            P24 = mean(P24),
            P25 = mean(P25),
            P26 = mean(P26),
            P27 = mean(P27),
            P28 = mean(P28),
            P29 = mean(P29),
            P30 = mean(P30),
            P31 = mean(P31),
            P32 = mean(P32),
            P33 = mean(P33),
            P34 = mean(P34),
            P35 = mean(P35),
            P36 = mean(P36),
            P37 = mean(P37),
            P38 = mean(P38),
            P39 = mean(P39),
            P40 = mean(P40),
            P41 = mean(P41),
            P42 = mean(P42),
            P43 = mean(P43),
            P44 = mean(P44),
            P45 = mean(P45),
            P46 = mean(P46),
            P47 = mean(P47),
            P48 = mean(P48),
            P49 = mean(P49),
            P50 = mean(P50),
            P51 = mean(P51),
            P52 = mean(P52),
            P53 = mean(P53),
            P54 = mean(P54))



# 변수 변환 데이터 크기 체크
df2 %>% 
  count(TIME) %>% 
  summarise(n = sum(n))


# 연속 공정에 따른 시간 차이를 반영
E1 <- df2[1:14]        #time 차이 4hr      #1열은 time열, 나머지 변수 
E2 <- df2[,c(1,15:19)] #time 차이 3hr
E3 <- df2[,c(1,20)]    #time 차이 2.5hr
E4 <- df2[,c(1,21:23)] #time 차이 1.5hr
E5 <- df2[,c(1,24)]    #time 차이 1.0hr
E6 <- df2[,c(1,25:51)] #time 차이 0.5hr
E7 <- df2[,c(1,52:55)] #time 차이 0.0hr


# 시간 보정 
E1$TIME <-  E1$TIME + 14400  # 4hr 14400
E2$TIME <-  E2$TIME + 10800  # 3hr 10800
E3$TIME <-  E3$TIME + 9000   # 2.5hr 9000
E4$TIME <-  E4$TIME + 5400   # 1.5hr 5400
E5$TIME <-  E5$TIME + 3600   # 1hr 3600
E6$TIME <-  E6$TIME + 1800   # 0.5hr 1800




#  1.3 품질데이터와 공정 데이터 합치기(결측치 제거)

# Target변수 시간 변환 
library(lubridate)

head(df1$TIME,3)
df1$TIME <- ymd_hms(df1$TIME)
df1$TIME <- with_tz(df1$TIME, "Asia/Seoul")
df1$TIME <- round_date(df1$TIME, "30 mins") 

# 데이터 합치기
H1 <- merge(df1,E1, by= "TIME")
H2 <- merge(H1,E2, by= "TIME")
H3 <- merge(H2,E3, by= "TIME")
H4 <- merge(H3,E4, by= "TIME")
H5 <- merge(H4,E5, by= "TIME")
H6 <- merge(H5,E6, by= "TIME")
H7 <- merge(H6,E7, by= "TIME")

# 예측을 위한 대표 GRADE만 추출 
product_list <- c("LC168", "LC175", "LC165","LC185", "LC385")

H7 <- H7 %>% 
  mutate(GRADE = ifelse(GRADE %in% product_list, GRADE, NA))

# 결측치 제거 
df0 <- H7 %>% 
  filter(!is.na(VM)) %>% 
  select(-P17)

# saveRDS(df0, file="df0")

# 실제 데이터는 324M데이터가 있어야 하는데, 과정상 부득이 일부 데이터로만 반영되어 있어서 아래 데이터 부터는 전처리된 데이터를 불러와서 진행하는 것으로 대체하겠습니다. 


# Step10 데이터 취득 및 전처리 실시  -------------------------------------------

# 전처리 도구 불러오기
library(dplyr);library(tidyr)
df0 <- readRDS("df0")  

# Step11 데이터 탐색  ----------------------------------------------------------

# 기술통계량
summary(df0)     
# 변수(feature)들의 개별 변수에 이상치 값이 관찰됨. 실제로는 처리하고 가야함. 


# 그래프 분석(변수간의 상관 분석)
df_g1 <- df0[,c(4:20)]
c1_cor <- cor(df_g1)

library(corrplot)
corrplot(c1_cor)  
# P9 vs P14,  P1 vs P10 공선성이 예상됨. 

df_g2 <- df0[,c(4,21:34)]
c2_cor <- cor(df_g2)

corrplot(c2_cor)    
#P22 VM과 음의 상관관계가 예상됨
#P27~P31 공선성이 높을 것으로 예상됨. 



df_g3 <- df0[,c(4,35:45)]
c3_cor <- cor(df_g3)

corrplot(c3_cor) 
# P33~42 다중공선성이 높을 것으로 예상


df_g4 <- df0[,c(4,46:57)]
c4_cor <- cor(df_g4)

corrplot(c4_cor) 
# P50, P52 VM과 상관관계가 높을 것으로 예상됨. 


# 상관관계가 관찰된 변수들을 추가 그래프 분석 진행함. 
df0 %>% 
  select(VM, P22, P26, P27, P50, P52) %>% 
  pairs()


# 상관관계 통계적 분석 
cor.test(df0$VM, df0$P22)
cor.test(df0$VM, df0$P50)


df0 %>% 
  select(VM, P27, P28, P29, P30, P31) %>% 
  pairs()


# VM과 P27의 상관관계 분석 
cor.test(df0$VM, df0$P27)



# 시계열 VM 경향 확인 
library(ggplot2)
ggplot(df0, aes(TIME, VM))+
  geom_point() + 
  scale_x_datetime(date_breaks = "2 month", 
                   date_labels = "%y %m/%d")




# 데이터 요약
summary(df0)
colnames(df0)
df <- df0[,4:57]  # 변수 제외 Time, GRADE, MI복사본 만들기
colnames(df)
dim(df)


# Tip. missing value를 쉽게 찾는 방법 -----
# install.packages("VIM")
library(VIM)  # missing value 찾는 방법 
aggr(df)
# install.packages("naniar")
library(naniar)  # missing value 찾는 방법 
vis_miss(df)
gg_miss_var(df)
df <- df %>% select(-P32)
gg_miss_var(df)
df <- df %>% select(-P18)
gg_miss_var(df)

dim(df)
head(df)



# nearZero분산 인자 제거하는 함수 
library(caret)
library(mlbench)
nearZeroVar(df)
df <- df[,-nearZeroVar(df)]  # 51번 항목 nearZero값으로 제거함. 




# 전처리 완료된 데이터 셋을 활용해서 과정을 진행해보세요.  --------------------
# Step12 핵심인자 선정  --------------------------------------------------------
## 데이터 Set 구분하기
# saveRDS(df,file="POE")
df <- readRDS("POE")  





## 데이터 Set 구분하기
str(df)
dim(df)
nrow(df)

set.seed(7279)
train=sample(nrow(df),nrow(df)*0.7)
train

test=(1:c(nrow(df)))[-train]
test

length(train)
length(test)


#df_train = df[train,] #교육 목적으로 의미 전달하여 보여줌. 
#df_test = df[test,] #교육 목적으로 의미 전달하여 보여줌. 

#head(df_train)
#head(df_test)


# Step13 분석 모형 검토  -------------------------------------------------------

## 분석실시(modeling) regression, rf
colnames(df)[1] = "y"
colnames(df)
lm.fit=lm(y~.,data=df[train,])

step(lm.fit)



lm.fit_best = lm(y ~ P4 + P7 + P9 + P11 + P12 + P13 + P14 + P20 + 
                   P23 + P26 + P30 + P31 + P34 + P35 + P37 + P39 + P45 + P46 + 
                   P47 + P54, data = df[train, ])



library(car)
vif(lm.fit_best)

#   P4         P7         P9        P11        P12        P13        P14        P20        P23 
# 3.990368   2.306867  20.825995   2.068444   2.186237  14.534759   5.082496   4.329208   5.430791 
#   P26        P30        P31        P34        P35        P37        P39        P45        P46 
# 3.274400 937.056587 915.533778   7.416290  27.314831  23.322396  11.066578   4.352388   1.356226 
#   P47        P54 
# 5.084212   1.814690 


# 엔지니어가 기술적 검토 시 아래와 같이 한개의 변수만 사용하는 것으로 함 (공선성 제거를 위함)
# P27~32는 Vacuum Unit 압력 센서로 동일환경을 대변하므로 1개만 사용
# P34~38는 압출기 온도로 동일환경을 대변하므로 1개만 사용
# P40~45는 Vacuum Dome 온도로 동일 환경을 대변함 1개만 사용
# P46~48는 전단압출기 압력으로 1개가 전체를 대변 가능함.  


# 이하 모든 분석에서는 아래의 변수를 제외하고 분석 진행. 
df_new <- df %>% select(-c(P28,P29,P30,P31,P35,P36,P37,P38,P41,P42,P43,P44,P45,P47,P48))
lm.fit=lm(y~.,data=df_new[train,])
step(lm.fit)



lm.fit_best = lm(y ~ P9 + P11 + P13 + P14 + P20 + P23 + P24 + P26 + 
                   P27 + P34 + P40 + P54, data = df_new[train,])
#최종 중요 변수 P9,P11,P13,P14,P20,P23,P24,P26,P27,P34,P40,P54

library(car)
vif(lm.fit_best)
#P9, P13에 약한 공선성이 관찰됨, 해석시 주의 

summary(lm.fit_best)
par(mfrow=c(2,2))
plot(lm.fit_best)
dev.off()

y_obs <- df[train,]$y
yhat_lm <- predict(lm.fit_best, newdata = df[train,])

library(DescTools)
RMSE(y_obs, yhat_lm)






# 리지, 라쏘 모형 
library(glmnet)
xx <- model.matrix(y~., df)

x <- xx[train,]
y <- df[train,]$y
glimpse(x)

set.seed(0319)
data_cvfit <- cv.glmnet(x, y, alpha = 1)
plot(data_cvfit) # 람다가 왼쪽에서 오른쪽 증가함에 따라 모수의 개수는 줄어들고 모형 간단

# alpha = 1.0(LASSO모형), alpha = 0.0(Ridge모형), alpha = 0.5 (ElasticNet 모형)


coef(data_cvfit, s = c("lambda.1se")) 
# 주요 변수  P10,P11,P14,P22,P23,P24,P26,P30,P34,P41,P45,P54

coef(data_cvfit, s = c("lambda.min")) #해석보단 예측이 초점일때 사용  
# 주요 변수  P3,P8,P10,P11,P13,P14,P21,P22,P23,P24,P26,P41,P45,P54


# predict(data_cvfit, s="lambda.1se", newx = x[1:5,])
predict(data_cvfit, s="lambda.min", newx = x[1:5,])  



y_obs <- df[train,]$y
yhat_glmnet <- predict(data_cvfit, s="lambda.min", newx=xx[train,])
yhat_glmnet <- yhat_glmnet[,1] #change to a vector from[n*1] matrix(1열 데이터만)




library(DescTools)
RMSE(y_obs, yhat_glmnet)



# 랜덤포레스트 
library(randomForest) ; library(tree)
set.seed(0319)
rf.fit=randomForest(y~.,data=df[train,],importance=T)
rf.fit

# importance(rf.fit)
varImpPlot(rf.fit)

# 회귀식에서 사전에 고려해서 동일 효과가 예상되는 조건에 대해서는 하나의 인자만 반영함. 
# P27~32는 Vacuum Unit 압력 센서로 동일환경을 대변하므로 1개만 사용
# P34~38는 압출기 온도로 동일환경을 대변하므로 1개만 사용
# P40~45는 Vacuum Dome 온도로 동일 환경을 대변함 1개만 사용
# P46~48는 전단압출기 압력으로 1개가 전체를 대변 가능함.  

set.seed(0319)
rf.fit_best=randomForest(y~P26+P27+P6+P11+P23+P29+P50+P24+P1+P10+P47+P39+P20+P34+P25+P38+P33+P22, data=df[train,],importance=T)
rf.fit_best
varImpPlot(rf.fit_best)

set.seed(0319)
rf.fit_best=randomForest(y~P26+P27+P29+P24+P6+P50+P10+P20+P11+P23+P25+P33+P47+P34, data=df[train,],importance=T)
rf.fit_best
varImpPlot(rf.fit_best)

set.seed(0319)
rf.fit_best=randomForest(y~P26+P27+P29+P24+P6+P50+P10+P20+P11+P23+P25+P47, data=df[train,],importance=T)
rf.fit_best
varImpPlot(rf.fit_best)

set.seed(0319)
rf.fit_best=randomForest(y~P26+P27+P29+P24+P6+P50+P10+P23+P25+P47, data=df[train,],importance=T)
rf.fit_best
varImpPlot(rf.fit_best)

# 중요도순서 P26,P27,P29,P24,P6,P50,P10,P23,P25,P47

y_obs <- df[train,]$y
yhat_rf <- predict(rf.fit_best, newdata = df[train,])

library(DescTools)
RMSE(y_obs, yhat_rf)

#회귀 중요 변수 P4,P5,P10,P11,P14,P16,P20,P22,P26,P27,P39,P47,P50,P51,P52
#RF   중요 변수 P26,P27,P29,P24,P6,P50,P10,P23,P25,P47
#라쏘 중요 변수 P3,P8,P10,P11,P13,P14,P21,P22,P23,P24,P26,P41,P45,P54



# ■ Improve ----------------------------------------------------------------------
# Step14 최적 모형 수립  -------------------------------------------------------

lm_obs = df[test,]$y #실제 관측값
lm_pred = predict(lm.fit_best,newdata=df[test,-1]) # 예측값 
rf_pred = predict(rf.fit_best,newdata=df[test,-1]) # 예측값 
yhat_glmnet <- predict(data_cvfit, s="lambda.min", newx=xx[test,])
glmnet_pred <- yhat_glmnet[,1] #change to a vector from[n*1] matrix


library(DescTools)
MSE(glmnet_pred, lm_obs)
MSE(lm_pred, lm_obs)
MSE(rf_pred, lm_obs)

(cor(glmnet_pred,lm_obs))^2  #상관계수 제곱    # 라쏘 모형이 가장 잘 맞음. 80.8%
(cor(lm_pred,lm_obs))^2  #상관계수 제곱        # 다중공선성이 있어 모형에서 제외
(cor(rf_pred,lm_obs))^2  #상관계수 제곱        # 교육 목적 상 설명력 77%을 최종 모형으로 선정 



# Step15 모형 검증 및 최적화  --------------------------------------------------


# 변수의 범위 확인 / 엔지니어링적인 관점에서 변수의 조절값은 검토해야함. 
# 아래는 Project Y가 망소특성을 만족하는 조건을 확인하는 것으로 마무리함. 
df_range <- df %>% select(P26,P10,P23,P27,P29,P24,P47,P6,P50,P25)

# xrange 함수를 임으로 만들어서 진행. 
xrange = function (x) 
{ print("Feature range")
  x1 = apply(x, 2, range)
  x2 = apply(x, 2, median)
  return(list(range = x1, median = x2))
}


xrange(df_range)

# [1] "Feature range"
# $range
# [1] "Feature range"
# $range
#           P26         P10      P23        P27        P29        P24       P47        P6       P50
# [1,] -0.95043457    1.517148 165.8004 -0.9347644 -0.9505098 -0.9025385 -1.181454 -3.764436  5.483384
# [2,]  0.07891881 4410.601833 245.7898 -0.3256397 -0.2966908  3.0526708 31.519875 70.048992 74.849144
#           P25
# [1,] -0.8933117
# [2,]  4.6584223



library(rpart)

df_new <- df[sample(nrow(df),nrow(df)),]
fit.dt <- rpart(y~P26+P27+P29+P24+P6+P50+P10+P23+P25+P47, data=df_new[train,])

library(rpart.plot)
rpart.plot(fit.dt)

new=data.frame(P26 = -0.95,  # 망소
               P27 = -9.3,   # 망소 
               P29 = -0.95,  # 망소
               P24 = -0.9,   # 망소 
               P6  = 1,      # 망대
               P50 = 5.48,   # 망소
               P10 = 1,      # 망대
               P23 = 245,    # 망대
               P25 = 4.65,   # 망대
               P47 = 31)     # 망대
                
         
#최적조건 구하기- 인자별 Range확인 (예측) 
predict(rf.fit_best,newdata=new)



# Step16 개선 결과 검증(Pilot Test) --------------------------------------------



# ■ Control --------------------------------------------------------------------
# Step17 최적모형 모니터링  ----------------------------------------------------
# Step18 표준화 및 수평전개  ---------------------------------------------------
# 이하 부록 --------------------------------------------------------------------





# 회귀식에 다중 공선성이 없다는 가정하에 회귀식을 최적화 할 경우 아래와 같이 해보세요. 

df_range <- df %>% select(P9,P11,P13,P14,P20,P23,P24,P26,P27,P34,P40,P54)

# xrange 함수를 임으로 만들어서 진행. 
xrange = function (x) 
{ print("Feature range")
  x1 = apply(x, 2, range)
  x2 = apply(x, 2, median)
  return(list(range = x1, median = x2))
}


xrange(df_range)

# [1] "Feature range"
# $range
#            P9          P11      P13       P14      P20      P23        P24         P26        P27
# [1,] 0.0006876684    0.6030414 25.54993 -43.81322 200.0173 165.8004 -0.9025385 -0.95043457 -0.9347644
# [2,] 9.7289481667 7450.1426000 57.76683 181.21718 270.2428 245.7898  3.0526708  0.07891881 -0.3256397
#          P34      P40      P54
# [1,] 191.9262 143.5439 14.90076
# [2,] 253.4755 211.9129 28.98650


  

summary(lm.fit_best) # 부호 방향 확인 

median(df$P11)
boxplot(df$P4)$stats[5]

# Project Y가 망소 특성이므로 회귀 계수의 부호 방향을 확인하고, 인자의 최적값 선택 
new=data.frame(P9 = 0.0006, # 회귀계수 부호 양의 값이므로,  변수값 최소값 선택
               P11 = 0.6,   #양
               P13 = 57.76, #음
               P14 = -43,   #양
               P20 = 200,   #양
               P23 = 245,   #음
               P24 = -0.9,  #양
               P26 = -0.95, #양
               P27 = -0.93, #양
               P34 = 253,   #음
               P40 = 211,   #음
               P54 = 28)    #음


predict(lm.fit_best,newdata=new)   
# 음의 값이 나와 현실성이 낮음, 변수의 범위를 엔지니어 관점에서 점검 필요. 



#_______________________----
# VCM----

# 과제 3 DX-LSS Simulation 실습과제 --------------------------------------------
# 작성자 :      
# Text encoding : UTF-8

# 세부 환경 설정 ---------------------------------------------------------------

dir()

# 아래 파일이 보이면 별도 Working directory를 재설정 하지 않으셔도 됩니다. 
# "k56.csv"   "VCM.R"     "VCM.Rproj"


# set Working directory 
# getwd()
setwd("D:/#.Secure Work Folder/DX-LSS-Project/DX-LSS MBB simulation/VCM")
# R Studio에서 Session > Set working directory > choose directory 로 설정 가능



# ■ Define ---------------------------------------------------------------------

# Step01 개선 기회 탐색  -------------------------------------------------------
# 정보화된 시스템 중 어느 시스템을 통해 개선 기회를 수시 점검 및 발굴 할 수 있는가?
# VCM 측정값도 엑셀관리, 환경이슈 점검 이슈는 별도 
# 과제명 : 폐수 운전 최적화를 통해 환경 이슈 개선 


# Step02 개선 기회 발굴 및 과제 선정--------------------------------------------
# - Big Y - little y 전개를 통해 과제 선정



# Step03 Project Y 선정  -------------------------------------------------------
## PVC 공장 폐수내 VCM 농도를 KPI로 선정
### 매년 PVC 2팀 공정 폐수 VCM배출량은 '18년 8676, 19년 3230으로 20년 548을 목표로하며, VCM함량 초과로 가동 중단 시
# 공정 전체 S/D이 되어야 하는 법적 Risk가 존재, 사외 배출 법적 기준 1ppm 미만으로 개선이 필요. 
### 평균 농도(ppm)  1.3 -> 0.5
### Max 농도(ppm) 45 -> 1
### 표준편차 3.1 -> 0.3



# ■ Measure ----------------------------------------------------------------------
# Step04 데이터 수집 및 검증 계획 수립   ---------------------------------------
# 폐수 Stripping Tower 후단 배출 폐수의 VCM 농도(1회/일)
# 측정 지표 : VCM함량(ppm), 수집 기간 : '20. 1~9월(1분기)


# Step05 데이터 Set 구성 -------------------------------------------------------
dir()
library(readr)
z = read_csv("k56.csv")

colnames(z)
str(z)
#View(z)
colnames(z)[1]=c("time")
colnames(z)[2]=c("y")

boxplot(z$y)
plot(z$time, z$y)

# dim(z)
z = na.omit(z)
dim(z)

# Step06 데이터 취득 시스템(유용성)검증  ---------------------------------------
# Step07 프로세스 현수준 파악  -------------------------------------------------
#Install.packages(SixSigma)
library(SixSigma)
#ss.study.ca(xST, xLT = NA, LSL = NA, USL = NA, Target = NA)
ss.study.ca(xST=z$y, USL = 0.5, Target = 1.0)


## project Y인 폐수 Stripping Tower 후단 배출 폐수의 VCM농도 CTQ로 선정함. 


# Step08 개선 목표 설정  -------------------------------------------------------
# VCM 평균 0.32ppm -> 평균 0.1ppm 이하
# Vcm 표준편차 0.75 -> 0.3이하


# ■  Analyze ----------------------------------------------------------------------
# Step09 X인자 검증 계획 수립  -------------------------------------------------

# 아래부터는 완료된 코드가 아닙니다. 

# 데이터 수집 계획
# project Y : VM data - LIMS (1회/1일, 07:00)
# x's : 각 TAG data - PIS 공정 Data
# 데이터 Merge 필요 
colnames(z)
str(z)
head(z)

library(dplyr)
df <- z %>% mutate(P1 = as.numeric(P1),
                   P2 = as.numeric(P2),
                   P3 = as.numeric(P3),
                   P4 = as.numeric(P4),
                   P5 = as.numeric(P5),
                   P6 = as.numeric(P6),
                   P7 = as.numeric(P7),
                   P8 = as.numeric(P8),
                   P9 = as.numeric(P9),
                   P10= as.numeric(P10),
                   P11= as.numeric(P11),
                   P12= as.numeric(P12),
                   P13= as.numeric(P13),
                   P14= as.numeric(P14),
                   P15= as.numeric(P15),
                   P16= as.numeric(P16),
                   P17= as.numeric(P17),
                   P18= as.numeric(P18),
                   P19= as.numeric(P19),
                   P20= as.numeric(P20),
                   P21= as.numeric(P21),
                   P22= as.numeric(P22),
                   P23= as.numeric(P23),
                   P24= as.numeric(P24),
                   P25= as.numeric(P25),
                   P26= as.numeric(P26),
                   P27= as.numeric(P27),
                   P28= as.numeric(P28)
                   )
                   


str(df)

df2 = df
colnames(df2)


df2 = df2[,c(1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58)] #변수 선택
str(df2)

df = df2

str(df)     
summary(df)
                   

# Step10 데이터 취득 및 전처리 실시  -------------------------------------------

# 전처리 도구 불러오기
library(dplyr);library(tidyr)


# Step11 데이터 탐색  ----------------------------------------------------------
#graph분석
pairs(df[,-1])  #time 열을 제외하기 위해 -1입력 함     *df[,-1] 의미 데이터셋[행,열]

#
boxplot(df$y) #상자 그림 그려주는 함수 


boxplot.stats(df$y) #상자 그림의 요소 보여줌

boxplot.stats(df$y)$stats[1]   #Q1-1.5*IQR
boxplot.stats(df$y)$stats[5]   #Q3+1.5*IQR 



df2 = df %>% filter(df$y>boxplot.stats(df$y)$stats[1], df$y<boxplot.stats(df$y)$stats[5])
df = df2


# 데이터 요약
summary(df)



# 전처리 완료된 데이터 셋을 활용해서 과정을 진행해보세요.  --------------------
# Step12 핵심인자 선정  --------------------------------------------------------
## 데이터 Set 구분하기
# saveRDS(df,file="k56")
df <- readRDS("k56")  

str(df)
dim(df)
nrow(df)
set.seed(7279)
train=sample(nrow(df),nrow(df)*0.7)
train

test=(1:c(nrow(df)))[-train]
test

length(train)
length(test)

# df_train = df[train,]
# df_test = df[test,]
# 
# head(df_train)
# head(df_test)


# Step13 분석 모형 검토  -------------------------------------------------------



# ■ Improve ----------------------------------------------------------------------
# Step14 최적 모형 수립  -------------------------------------------------------
## 분석실시(modeling) regression, rf
lm.fit=lm(y~.,data=df[train,-1])
step(lm.fit)
lm.fit_best = lm(y ~ P5 + P10 + P22 + P27 + P28, data=df[train,-1])

library(car)
vif(lm.fit_best)


set.seed(0319)
library(randomForest) ; library(tree)
rf.fit=randomForest(y~.,data=df[train,-1],importance=T)
rf.fit
# importance(rf.fit)
varImpPlot(rf.fit)


# 랜덤포레스트 주요 인자 기준으로 모형 재수립
set.seed(0319)
rf.fit_best=randomForest(y~P5+
                           P22+
                           P8+
                           P6+
                           P3,data=df[train,-1],importance=T)
rf.fit_best
# importance(rf.fit)
varImpPlot(rf.fit_best)



# Step15 모형 검증 및 최적화  --------------------------------------------------

lm_obs = df[test,]$y #실제 관측값
lm_pred = predict(lm.fit_best,newdata=df[test,-1]) # 예측값 
rf_pred = predict(rf.fit_best,newdata=df[test,-1]) # 예측값 

library(DescTools)
MSE(lm_pred, lm_obs)
MSE(rf_pred, lm_obs)

RMSE(lm_pred, lm_obs) # 회귀식이 상대적으로 우수함
RMSE(rf_pred, lm_obs)

(cor(lm_pred,lm_obs))^2  #상관계수 제곱
(cor(rf_pred,lm_obs))^2  #상관계수 제곱

df_new <- df %>% select(P5,P22,P8,P6,P3,P25,P9,P1,P27,P28) #P10은 정수 

print("Feature range")
for( i in 1:ncol(df_new)){
  A = df_new %>% filter(df_new[,i]>0) %>% .[,i] 
  B = A[A>boxplot(A)$stats[1]&A<boxplot(A)$stats[5]] %>% range()
  print(data.frame(names=colnames(df_new)[i] ,lower=B[1],upper=B[2]))
}

table(df$P10)

summary(lm.fit_best) # 부호 다시 체크,  project Y 망소 특성

new=data.frame(P5= 72644,    #음 (72644~292921)
               P10= 1.3,      #음 (1.3, 1.4, 4.6, 5, 5.6, 6)
               P22= 0.08,    #음 (0.0856 ~ 21.4116)
               P27= 5.5175 , #양 (4.7114 ~ 5.5175)
               P28= 6.76       #음 (6.7655 ~ 50.0666)
               )


new





#최적조건 구하기- 인자별 Range확인 (예측) 
predict(lm.fit_best,newdata=new)


#최적조건 추가 확인_randomForest

library(rpart)
tr.fit = rpart(y~P5+
                 P22+
                 P8+
                 P6+
                 P3,data=df[train,-1])

library(rpart.plot)
rpart.plot(tr.fit)


library(dxlss) #범수의 범위  평균값 보여주는 함수
df_new2 <- df %>% select(P5,P22, P8, P6, P3) 
step3(df_new2)


boxplot(df$LIC9106.PV)$stats
new=data.frame(P5  = 293779,  # 양 145000이상(68359~293779)
               P22 = 21.4,    # 양 0.38이상(-2.89~67.59)
               P8  = 15.9,    # 평균(값 조정 확인)
               P6  = 40 ,     # 양 34이상(23.128~40.2) 
               P3  = 66.7)   # 평균(값 조정 확인)  


predict(rf.fit_best,newdata=new)


# Step16 개선 결과 검증(Pilot Test) --------------------------------------------

# 상기 조건으로 pilot Test 진행 후 결과 체크 


# ■ Control ----------------------------------------------------------------------
# Step17 최적모형 모니터링  ----------------------------------------------------

# Step18 표준화 및 수평전개  ---------------------------------------------------

#_______________________----
# tb (터보 블로워) ----


# 과제 4 DX-LSS Simulation 실습과제 --------------------------------------------
# 작성자 :      
# Text encoding : UTF-8


# 세부 환경 설정 ---------------------------------------------------------------

dir()  #아래와 같이 뜨면 Working Directoring 재설정 불필요함. 

# [1] "20201024.xls" "20201025.xls" "20201026.xls" "20201027.xls" "20201028.xls"
# [6] "20201029.xls" "20201030.xls" "20201031.xls" "20201101.xls" "20201104.xls"
# [11] "20201105.xls" "20201106.xls" "20201107.xls" "20201108.xls" "20201109.xls"
# [16] "20201110.xls" "20201111.xls" "20201112.xls" "20201113.xls" "20201115.xls"
# [21] "20201116.xls" "20201117.xls" "20201118.xls" "20201119.xls" "20201120.xls"
# [26] "20201121.xls" "20201122.xls" "20201123.xls" "20201124.xls" "20201125.xls"
# [31] "20201126.xls" "20201127.xls" "20201128.xls" "20201129.xls" "20201130.xls"
# [36] "20201201.xls" "20201202.xls" "20201203.xls" "20201204.xls" "20201205.xls"
# [41] "20201206.xls" "20201207.xls" "20201208.xls" "20201209.xls" "20201210.xls"
# [46] "20201211.xls" "20201212.xls" "20201213.xls" "20201214.xls" "20201215.xls"
# [51] "20201216.xls" "tb.R"         "tb.Rproj"  


# set Working directory 
# getwd()
setwd("D:/#.Secure Work Folder/DX-LSS-Project/DX-LSS MBB simulation/tb")
# R Studio에서 Session > Set working directory > choose directory 로 설정 가능



# ■ Define ---------------------------------------------------------------------

# Step01 개선 기회 탐색  -------------------------------------------------------
# - 시스템에서 과제 도출 가능? 정보화 시스템을 통해서 과제 도출 가능함.  

# Step02 개선 기회 발굴 및 과제 선정--------------------------------------------
# - Big Y - little y 전개를 통해 과제 선정?  다수 과제 도출되었으며, 직접 
#   다룰 수 있는 과제를 최종 선정함. 


# Step03 Project Y 선정  -------------------------------------------------------
## KPI  : 터보 블로워 가동율 최적 조건 도출 
## CTQ  : 폭기조#2 DO


# ■ Measure ----------------------------------------------------------------------
# Step04 데이터 수집 및 검증 계획 수립   ---------------------------------------


# Step05 데이터 Set 구성 -------------------------------------------------------
library(readxl)

# 10/25~12/16일 데이터 
# 동일형식의 데이터 한번에 불러오기  
# *주의사항 : 폴더 내에 형식이 다른 파일 존재하면 에러가 발생함. 
library(readxl)
dir <- ("D:\\#.Secure Work Folder\\DX-LSS-Project\\DX-LSS MBB simulation\\tb\\DAT")
file_list <- list.files(dir)

# 데이터 불러오기 및 합치기
data <- data.frame()

for(file in file_list) {
  print(file)
  temp <- read_excel(paste(dir, file, sep = "\\"), skip = 1 )
  data <- rbind(data,temp)
}


# 일반적으로 경로 하위 원화 표시 '\'를 사용하려면 두개 넣어야함 '\\'
# 그래서 다른 방법으로 백슬러시를 활용하면 됨 '/'  입력키는 Shift key옆 '?/' key

library(readxl)  
dir <- ("D:/#.Secure Work Folder/DX-LSS-Project/DX-LSS MBB simulation/tb/DAT")
file_list <- list.files(dir)

# 데이터 불러오기 및 합치기
data <- data.frame()

for(file in file_list) {
  print(file)
  temp <- read_excel(paste(dir, file, sep = "/"), skip = 1 )
  data <- rbind(data,temp)
}


z <- data  #복사본 만들기

head(z)
str(z)
colnames(z)
head(z$TIME)
tail(z$TIME)
# TIME , FT-102=f1, PH-103=f2, 온도=tmp, do=DO, 가동률=oper, MLSS=mlss, TOC...8 = toc1, TOC...9 = toc2,  TN = TN  으로 전환 

#rename함수를 활용해도 됨
colnames(z)[1:10]=c("time", "f1","f2","tmp","do","oper","mlss","toc1","toc2","TN")
colnames(z)[5]=c("y")   #DO농도를 y로 명칭 변경함 
colnames(z)

dim(z)
dim(na.omit(z))

head(z)
colnames(z)
library(gridExtra)
library(dplyr)
library(ggplot2)
data <- z[,-1]
summary(data)
str(data)

boxplot(data$y)$stats
boxplot(data$mlss)$stats
data <- data %>% filter(y <= 5 , mlss > 4452)

glimpse(data)



panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}


# pairs(data %>% sample_n(min(1000, nrow(data))))
pairs(data %>% sample_n(min(1000, nrow(data))),
      lower.panel=function(x,y){ points(x,y); abline(0, 1, col='red')},
      upper.panel = panel.cor)




# 엑셀 데이터에 2줄이상 Colname이 있는 경우 가끔 날짜를 숫자로 변환됨
# 이를 해결하기 위해서는 날짜 열에 대해서 강제 변환이 되는 오류를 막기 위해
# 열 이름의 행수를 제한하면 개선됨 .  

# library(lubridate)
# time <- z$time
# parse_date_time(time, orders = c('ymdHMS') )

z[,1] = as.POSIXct(z$time)#LOGGING - 시간에 따른 변화 체크 변환이 안됨
z[,2] = as.numeric(z$f1) # 유입수 펌프 - 토출유량 #조절 가능 함
z[,3] = as.numeric(z$f2) # 유입수 토출 -  # 
z[,4] = as.numeric(z$tmp) # 폭기조#1 - 폭기조 온도         #조절 가능 
z[,5] = as.numeric(z$y) # 폭기조#2 - 폭기조 DO         #DO를 일정 수준으로유지하고 싶음.
z[,6] = as.numeric(z$oper) # 폭기조 블로워 - 가동률     #가동을 어떻게 해야하는지 알고 싶은 X임
z[,7] = as.numeric(z$mlss) # 폭기조#3 - 폭기조 MLSS    # DO에 영향 미치고, 조절 가능한 항목
z[,8] = as.numeric(z$toc1) # 원수 - (TOC)     #조절 불가
z[,9] = as.numeric(z$toc2) # 방류수 -  (TOC)   #조절 불가 
z[,10] = as.numeric(z$TN) # 방류수 - (TN)     #조절 불가 


#폭기조 블로워 가동률에 대한 중요 변수로 고려하고 있음. 5,6번 영향성을 중요하게 보고 있음. 

# Step06 데이터 취득 시스템(유용성)검증  ---------------------------------------
dim(z)
z1 = z  # 복사본 만들기 
z1 = na.omit(z1) # 결측치 제거
dim(z1) # 결측치가 없어서 그대로  데이터 셋명 z를 그대로 이용함. 
# z = z1

boxplot(z$y)
boxplot.stats(z$y)$stats[5]
boxplot.stats(z$y)$stats

library(dplyr)  #통계적 방법을 통한 이상치 제거
z = z %>% 
  filter(boxplot.stats(z$y)$stats[1]<z$y, 
         boxplot.stats(z$y)$stats[5]>z$y)
# z1 = z %>% filter(2.525<z$y, 5.9375>z$y)
dim(z)


head(z)
hist(z$y)

plot(z$time, z$y)
hist(z$y)
boxplot(z$y)


# Step07 프로세스 현수준 파악  -------------------------------------------------

#rm(list=ls())
#dev.off()

## project Y는 DO로 선정함

## DO 선정 배경
### 1. 분석 시간이 짧아 피드백이 빠르며, 공정 제어 지표로서 적절함
### 2. 생산 제품의 현재 휘발분 함량 상태를 즉각적으로 대변함
### 3. 동일주기 측정으로 (매일 07:00) 공정 상태와의 Data 연계가 보다 용이하여 예측 모델의 신뢰성 확보 가능

# DO 수집 방법 HMI 시스템(실시간, 주기:?)


#Install.packages(SixSigma)
library(SixSigma)
#ss.study.ca(xST, xLT = NA, LSL = NA, USL = NA, Target = NA)
ss.study.ca(xST=z$y, LSL = 1, USL = 4, Target = 2)



# Step08 개선 목표 설정  -------------------------------------------------------
## DO 평균 5.6ppm(Z bench : -1.9) -> 평균 2ppm 이하(Z bench )



# ■  Analyze ----------------------------------------------------------------------
# Step09 X인자 검증 계획 수립  -------------------------------------------------

# 데이터 수집 계획
# project Y : DO data - HMI시스템 (매 10분당)

# x's : HMI system, 기온 기상청 데이터, 폭기조내 MLSS 농도는 현재 미측정
# 최종 data set 구성 필요. 

# Measure단계에 처리하여 별도 처리 없음 



# Step10 데이터 취득 및 전처리 실시  -------------------------------------------

# 전처리 도구 불러오기
library(dplyr);library(tidyr)


# Step11 데이터 탐색  ----------------------------------------------------------

# 데이터 요약
df = z
dim(df)
df <- na.omit(df)
dim(df)
summary(df)

df = df %>% 
  filter(0<df$toc2 & 0<df$f1)
dim(df)

df = df %>% 
  filter(boxplot.stats(df$TN)$stats[1]<df$TN, 
         boxplot.stats(df$TN)$stats[5]>df$TN)

boxplot(df$f1)



#graph분석

# df_cor <- df %>% select(-time)
df2 <- df %>% select(y, f1, f2, tmp, oper, mlss, toc1, toc2, TN)
df_cor <- cor(df2)
df_cor

library(corrplot)
corrplot(df_cor)   

# 해당 부분은 옵션으로 별도의 교육은 없이 진행.   
library(ggplot2)
library(dplyr)
colnames(df)

df %>% ggplot(aes(time,y))+geom_point() +
  scale_x_datetime(date_breaks = "3 day", date_labels = "%m/%d")
 
# 시간에 따른 경향성 변수 경향성 체크  
df %>% ggplot(aes(time,y))+geom_point(aes(col=tmp))
df %>% ggplot(aes(time,y))+geom_point(aes(col=f1))
df %>% ggplot(aes(time,y))+geom_point(aes(col=f2))
df %>% ggplot(aes(time,y))+geom_point(aes(col=mlss))
df %>% ggplot(aes(time,y))+geom_point(aes(col=TN))
df %>% ggplot(aes(time,y))+geom_point(aes(col=toc1))
df %>% ggplot(aes(time,y))+geom_point(aes(col=toc2)) 


# boxplot을 위해서 가동률을 Factor 데이터로 변환하여 확인 
df_new = df
df_new$oper = as.factor(df_new$oper)

df_new %>% ggplot(aes(oper,y)) + geom_jitter(aes(col=tmp)) + geom_boxplot(alpha=0.3)
df_new %>% ggplot(aes(oper,y)) + geom_jitter(aes(col=f1)) + geom_boxplot(alpha=0.3)
df_new %>% ggplot(aes(oper,y)) + geom_jitter(aes(col=f2)) + geom_boxplot(alpha=0.3)
df_new %>% ggplot(aes(oper,y)) + geom_jitter(aes(col=mlss)) + geom_boxplot(alpha=0.3)
df_new %>% ggplot(aes(oper,y)) + geom_jitter(aes(col=TN)) + geom_boxplot(alpha=0.3)


# 전처리 완료된 데이터 셋을 활용해서 과정을 진행해보세요. ------
# Step12 핵심인자 선정  --------------------------------------------------------
## 데이터 Set 구분하기

# rm(list=ls())
# saveRDS(df, file = "df")
df <- readRDS("df")

# 복사본 만들기
df_temp <- df

str(df)
dim(df)
nrow(df)
set.seed(7279)
train=sample(nrow(df),nrow(df)*0.7)
train

test=(1:c(nrow(df)))[-train]
test

length(train)
length(test)

df_train = df[train,]
df_test = df[test,]

head(df_train)
head(df_test)



# Step13 분석 모형 검토  -------------------------------------------------------

# 아래는 각 인자별 관계를 확인하지 않고 그냥 DO가 Project Y 나머지를 그냥 기존 데이터로 생각하고 돌린 부분입니다. 이와 비슷하게 정리하려고 하오니, 해석에 오해가없도록 참조만 하세요. 
## 분석실시(modeling) regression, rf

set.seed(7279)
lm.fit=lm(y~.,data=df[train,-1])
step(lm.fit)
lm.fit_best = lm(y ~ f1 + f2 + tmp + oper + toc1 + TN, data = df[train, -1])

# 하지만 TN, toc1, toc2는 조절 불가 따라서 변수에서 삭제, 공변량으로 넣는 것은 체크 
lm.fit_best = lm(y ~ f2 + tmp + oper + mlss, data = df[train, -1])
summary(lm.fit_best)

# 도출식 y = 0.0239f2  +  -0.2579tmp  +  -0.0656oper  +  -0.00009601mlss 
# f2 유입수 토출, tmp 폭기조 온도, oper 폭기조 블로워(가동율), mlss 폭기조 MLSS 

library(randomForest) ; library(tree)
set.seed(7279)
rf.fit=randomForest(y~.,data=df[train,-1],importance=T)
rf.fit
# importance(rf.fit)
varImpPlot(rf.fit)

# TN, toc1, toc2는 조절 불가 따라서 변수에서 삭제
rf.fit_best=randomForest(y~ f2 + oper + tmp,data=df[train,-1],importance=T)
varImpPlot(rf.fit_best)



# ■ Improve ----------------------------------------------------------------------
# Step14 최적 모형 수립  -------------------------------------------------------

lm_obs = df[test,]$y #실제 관측값
lm_pred = predict(lm.fit_best,newdata=df[test,-1]) # 예측값 
rf_pred = predict(rf.fit_best,newdata=df[test,-1]) # 예측값 

library(DescTools)
MSE(lm_pred, lm_obs)
MSE(rf_pred, lm_obs)

RMSE(lm_pred, lm_obs) 
RMSE(rf_pred, lm_obs)

(cor(lm_pred,lm_obs))^2  #상관계수 제곱   # 설명력 25%
(cor(rf_pred,lm_obs))^2  #상관계수 제곱   # 설명력 94%



# Step15 모형 검증 및 최적화  --------------------------------------------------

colnames(df)
df_range = df %>% select(y, f1, f2, tmp, oper, mlss)
dim(df_range)
head(df_range)

print("Feature range")
for( i in 1:ncol(df_range)){       # 1:6에서 6은 변수의 숫자이며, 사용자에 맞게 변경 
  A = df_range %>% filter(df_range[,i]>0) %>% .[,i] 
  B = A[A>boxplot(A)$stats[1]&A<boxplot(A)$stats[5]] %>% range()
  print(data.frame(names=colnames(df_range)[i] ,lower=B[1],upper=B[2]))
}


library(dxlss)
str(df)
step3(df[,-1])

summary(lm.fit_best) # 부호 다시 체크 

new_lm=data.frame(f2   = 11.83962 ,  #양
                  tmp  = 31.6562 ,   #음
                  oper = 45,         #음
                  mlss = 4516)       #음


predict(lm.fit_best,newdata=new_lm)  #최적값에 대한 회귀모형 DO예측값은 3.47ppm


library(tree)  # 교재 중심 
head(df)
df_tr = tree(y~ f2 + tmp + oper + f1, data=df[test,-1])
plot(df_tr) ; text(df_tr)

library(rpart) # 다른 의사결정 나무 
df_tr_r = rpart(y~ f2 + tmp + oper + f1, data=df[test,-1])
library(rpart.plot)
rpart.plot(df_tr_r)
# 가동율이 가장 큰 분류 48이상, tmp 35이상, f2 12이하 조건에서 유리  

new_rf=data.frame(oper = 56,    #범위 45~56, 조건 48이상
                  f2   = 8.9 ,  #범위 8.9~11.8, 조건 12이하 
                  tmp  = 35 ,   #범위 31.6~36.17,조건 35이상(가동율 48이상일 때)
                  f1   = 463)   #범위 351~463, 조건 406이상     


new_rf

#최적조건 구하기- 인자별 Range확인 (예측) 
# predict(lm.fit_best,newdata=new_lm)   
predict(rf.fit_best,newdata=new_rf)   # 2ppm 달성 가능 
# predict(rf.fit,newdata=new)


# Step16 개선 결과 검증(Pilot Test) --------------------------------------------


# ■ Control ----------------------------------------------------------------------
# Step17 최적모형 모니터링  ----------------------------------------------------

# Step18 표준화 및 수평전개  ---------------------------------------------------

#_______________________----
# HSB ----

# 과제 2 DX-LSS Simulation 실습과제 --------------------------------------------
# 작성자 :      
# Text encoding : UTF-8

# 세부 환경 설정 ---------------------------------------------------------------

# set Working directory 
getwd()
setwd("D:/#.Secure Work Folder/DX-LSS-Project/DX-LSS MBB simulation/HSB")
# R Studio에서 Session > Set working directory > choose directory 로 설정 가능

dir()


# ■ Define ---------------------------------------------------------------------

# Step01 개선 기회 탐색  -------------------------------------------------------

# 정보화된 시스템 중 어느 시스템을 통해 개선 기회를 수시 점검 및 발굴 할 수 있는가?

# Step02 개선 기회 발굴 및 과제 선정--------------------------------------------
# Y-y 전개? FMEA, QFD, Process Map 등 

# Step03 Project Y 선정  -------------------------------------------------------
## 히알루론산나트륨의 극한 점도를 KPI로 선정


# ■ Measure ----------------------------------------------------------------------
# Step04 데이터 수집 및 검증 계획 수립   ---------------------------------------
# Data 수집 계획
# 측정 지표 : QC 릴리즈 - 극한점도(%), 수집 시스템 :   , 수집 기간 : 18년 ~ 20년
# 변수 18ea (종배양 ~  흡착)


# Step05 데이터 Set 구성 -------------------------------------------------------
library(readxl)
HA = read_excel("HSB.xlsx", sheet = 'HSB7',  skip=5)

str(HA)

library(dplyr)
colnames(HA)

df <- HA #데이터 복사본 만들기 

#bz가 종속변수 : 극한점도
hist(df$bz)       
plot(df$bz)
boxplot(df$bz)
summary(df$bz)


# Step06 데이터 취득 시스템(유용성)검증  ---------------------------------------

# Step07 프로세스 현수준 파악  -------------------------------------------------
#Install.packages(SixSigma)
library(SixSigma)
#ss.study.ca(xST, xLT = NA, LSL = NA, USL = NA, Target = NA)
ss.study.ca(xST=df$bz, LSL =90, USL =120, Target = 105)


# Step08 개선 목표 설정  -------------------------------------------------------
## QC점도 평균 105%(Z bench 2.19) -> 평균 107ppm 이상

# ■  Analyze ----------------------------------------------------------------------
# Step09 X인자 검증 계획 수립  -------------------------------------------------
# 데이터 수집 계획
# project Y : bz data - LIMS
# x's : 공정 Data

# Step10 데이터 취득 및 전처리 실시  -------------------------------------------
# 전처리 도구 불러오기
library(dplyr);library(tidyr)
# 데이터 Merge 불필요, 사전 데이터 전처리 완료 데이터 활용

# Step11 데이터 탐색  ----------------------------------------------------------

# 데이터 요약
summary(df)

#graph분석
df2 <- df %>% select(-a)
df_cor <- cor(df2)
df_cor

library(corrplot)
corrplot(df_cor)  


# 통계적 이상치 제거
dim(df)
df2  <-  df %>% filter(df$y>boxplot.stats(df$y)$stats[1], #Q1-1.5*IQR 이하 제거
                       df$y<boxplot.stats(df$y)$stats[5]) #Q3+1.5*IQR 이상 제거 
df <-  df2

dim(df)

# boxplot(df$bz) #상자 그림 그려주는 함수 
# boxplot.stats(df$bz) #상자 그림의 요소 보여줌
# boxplot.stats(df$bz)$stats[1]   #Q1-1.5*IQR
# boxplot.stats(df$bz)$stats[5]   #Q3+1.5*IQR 



# 전처리 완료된 데이터 셋을 활용해서 과정을 진행해보세요. ------
# saveRDS(df, file = "df")
df <- readRDS("df")

# Step12 핵심인자 선정  --------------------------------------------------------
## 데이터 Set 구분하기
str(df)
dim(df)
nrow(df)
set.seed(7279)
train=sample(nrow(df),nrow(df)*0.7)
test=(1:c(nrow(df)))[-train]


# length(train)
# length(test)
 
df_train = df[train,]
df_test = df[test,]
 
# head(df_train)
# head(df_test)




# Step13 분석 모형 검토  -------------------------------------------------------

## 분석실시(modeling) regression, rf
set.seed(7279)
lm.fit=lm(bz~.,data=df_train)
lm.fit=lm(bz~.-a,data=df_train)
step(lm.fit)
lm.fit_best = lm(bz ~ m + s + u + x + z + an + ao + bd + be, data = df_train)


library(randomForest) ; library(tree)
set.seed(7279)
rf.fit=randomForest(bz~.-a,data=df_train,importance=T)
rf.fit
# importance(rf.fit)
varImpPlot(rf.fit)

# 랜덤포레스트 주요 인자 기준으로 모형 재수립
rf.fit_best=randomForest(bz ~ ao +
                           y +
                           x + 
                           an +
                           s +
                           z +
                           m +
                           ak +
                           r +
                           bd, data=df_train,importance=T)


rf.fit_best
# importance(rf.fit)
varImpPlot(rf.fit_best)
# 중요도 순서 
# ao : 한외여과 공정 - 분리모드 전환 후 총 분리시간(h) 
# y  : 균체 제거 공정 - Recovery(회수) 시간(min)
# x  : 균체 제거 공정 - 균체 제거 시간(min)
# an : 한외여과 공정 - 순환 시간 흐름 시간 13.5초 이하 도달하기까지 시간 (min)



# ■ Improve ----------------------------------------------------------------------
# Step14 최적 모형 수립  -------------------------------------------------------

lm_obs = df_test$bz #실제 관측값
lm_pred = predict(lm.fit_best,newdata=df_test) # 예측값 
rf_pred = predict(rf.fit_best,newdata=df_test) # 예측값 
  
library(DescTools)
MSE(lm_pred, lm_obs)
MSE(rf_pred, lm_obs)

RMSE(lm_pred, lm_obs)  # 4.28 
RMSE(rf_pred, lm_obs)  # 2.8  

(cor(lm_pred,lm_obs))^2  #상관계수 제곱   # 설명력 62%
(cor(rf_pred,lm_obs))^2  #상관계수 제곱   # 설명력 78%



# Step15 모형 검증 및 최적화  --------------------------------------------------

# 모델링에 활용된 인자 범위 체크 
  # 불필요 변수 제거 
df_range <- df_test %>% 
  select(-a)

length(df_range) # for (i in 1: xx)   xx에 숫자를 length 값으로 넣어줌

print("Feature range")
for( i in 1:ncol(df_range)){
  A = df_range %>% filter(df_range[,i]>0) %>% .[,i] 
  B = A[A>boxplot(A)$stats[1]&A<boxplot(A)$stats[5]] %>% range()
  print(data.frame(names=colnames(df_range)[i] ,lower=B[1],upper=B[2]))
}

summary(lm.fit_best) # 부호 다시 체크,  project Y 망소 특성

#y = -2.26m  +  -1.667s  +  -2.08u  +  0.10x   +  -0.05z   +   -0.69an
#    +  3.799ao   +  1.57bd    -0.03be

# 최적 조건 : 전체 데이터 기준 
new=data.frame(m = 17.1,  # 음 / 종배양 3(h)
               s = 16.4,  # 음 /        
               u = 2,     # 음 / 균체제거 공정 - 회수 후 균체 제거 전 Holding시간
               x = 72,    # 양 / 균체 제거 공정 - 균체 제거 시간(min)
               z = 700,   # 음 / 균체 제거 공정 - Recovery 정체수 부피 (L)
               an = 105,  # 음 / 순환 시간 흐름 시간 13.5 이하 도달까지 시간(min)
               ao = 12.3, # 양 / 분리 모드 전환 후 총 분리 시간(h)
               bd = 11.6, # 양 / 흡착 공정 - 흡착 시간(h)
               be = 2342) # 음 / 흡착 공정 - 흡착제 제거 후 부피 (L)

new



#최적조건 구하기- 인자별 Range확인 (예측) 
predict(lm.fit_best,newdata=new)


#최적조건 추가 확인_randomForest
tr.fit = tree(bz ~ ao +
                y +
                x + 
                an +
                s +
                z +
                m +
                ak +
                r +
                bd,data=df_train)

plot(tr.fit) ; text(tr.fit)

# ao 11.15 이상, y 121.5 이상, x61 이상 

library(rpart)
df %>% head
tree_HA = rpart(bz ~ ao +
                  y +
                  x + 
                  an +
                  s +
                  z +
                  m +
                  ak +
                  r +
                  bd,data=df_train)

library(rpart.plot)
rpart.plot(tree_HA)

# ao 11 이상, y 122 이상 


varImpPlot(rf.fit) # ao 11.15 이상, y 121.5 이상, x61 이상 


# 최적 조건 도출을 위해 인자의 조절 범위는 분석에 사용한 범위 내에서 찾아야 하며, 엔지니어링 관점에서 메커니즘이 설명되고, 경제성을 고려해야함. 

new_rf=data.frame(ao = 12.3, # 양
                  y = 203,   # 양
                  x = 72, # 양   #이하는 회귀분석 시 도출된 최적값 반영 
                  m = 17.1, # 음 
                  s = 16.4, # 음  
                  u = 2, # 음
                  z = 700, # 음
                  an = 105, # 음
                  bd = 11.6, # 양
                  be = 2342, # 음
                  ak = 2170, # 임의값(1860~2170) 
                  r = 1185 ) # 임의값(1146~1185)



#최적조건 비교
predict(lm.fit_best,newdata=new)      #126%
predict(rf.fit_best,newdata=new_rf)   #113%


# Step16 개선 결과 검증(Pilot Test) --------------------------------------------
# 상기 조건으로 pilot Test 진행 후 결과 체크 


# ■ Control ----------------------------------------------------------------------
# Step17 최적모형 모니터링  ----------------------------------------------------

# Step18 표준화 및 수평전개  ---------------------------------------------------

