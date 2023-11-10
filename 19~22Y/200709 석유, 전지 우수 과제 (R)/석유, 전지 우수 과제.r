# ___________________________--------
# **************************************--------
# ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁====
# 01 _ 200709 석유, 전지 우수 과제----
# ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁====
# ___________________________--------  
# **************************************--------

# 석유 우수 사례 소개 ----

# DX-LSS MBB과정 ------------------------------------------------------------

# 작성자 : 백인엽P,  최종 수정일 : 20. 7. 6
# Text encoding : UTF-8

# 01. 세부 환경 설정 -----------------------------------------------------------

# set Working directory 

getwd()
setwd("D:/#.Secure Work Folder/DX-LSS-Project/DX-LSS강사과정/2020강사과정/석유")
# R Studio에서 Session > Set working directory > choose directory 로 설정 가능

dir()
rm(list = ls())   # Data set 삭제(Global Environment)


# 02. 데이터 전처리 ------------------------------------------------------------

# 전처리 도구 불러오기
library(dplyr);library(tidyr)

# 데이터 불러오기 
pis_product = read.csv("pis_raw_제품2.csv")
lims=read.csv("LIMS.csv") 
bucket=read.csv("Bucket.csv")

# 데이터 체크
head(pis_product)
head(lims)
head(bucket)


# lims 데이터 중 필요한 데이터만 선택
colnames(lims)
lims = lims %>%  select(Date.Time, PSD..LT.0.15mm., PSD..LT.0.85mm., PSD..0.85.2.0mm.)
# lims 데이터 중 project Y만 선택 Date.Time, PSD..LT.0.15mm., PSD..LT.0.85mm., PSD..0.85.2.0mm.
head(lims) # 변수 선택 저장 결과 체크


# D50 계산식 추가
str(lims) # PSD.. 0.85. 2. 0mm가 Factor로 들어가있으니, numeric으로 변경
lims[,4] = as.numeric(as.character(lims[,4]))  

# Factor는 반드시 character로 변환 후 numeric으로 변환시켜야 함.
# (아니면 다른 값이 됨)



str(lims) #변환 결과 체크 




# D50을 구하는 공식은 식은 내부에서 공식 활용
# D50 = 850 + ( 50 - 'PSD(LT 0.85mm)' ) / 'PSD(0.85~2.00)' X (2000-850)

lims=lims %>% mutate(D50=(850+(50-PSD..LT.0.85mm.)/PSD..0.85.2.0mm.*(2000-850))) 
head(lims) # 변환 결과 확인

lims=lims %>% .[,c(1,2,5)] # 사용할 변수만 저장. select 명령을 사용해도 됨 
head(lims)
summary(lims)


# data 현수준 파악을 위한 데이터 저장이며, R에서 현수준 파악해도 됨
write.csv(lims, 'lims_D50.csv')  # lims dataset만 저장됨
write.csv(bucket, 'buc.csv')
dir()

save(lims, file = 'lims')  # lims 데이터 셋 저장
save(bucket, file = 'bucket')  # bucket 데이터 셋 저장
save(pis_product, file = 'pis_product')  # pis_product 데이터 셋 저장

rm(list=ls())  #데이터셋 모두 삭제
dir()
load('lims') # 데이터 셋 불러오기 
load('bucket') # 데이터 셋 불러오기
load('pis_product') # 데이터 셋 불러오기



#날짜 데이터 변환하기 
pis_product$Date.Time = as.POSIXct(pis_product$Date.Time)
str(pis_product)
lims$Date.Time = as.POSIXct(lims$Date.Time)
str(lims)


# 날짜 형식 전처리  
head(bucket)
colnames(bucket)
colnames(bucket)[1]=c("Date.Time")
bucket[,'Date.Time']=paste0(20,bucket$Date.Time)
bucket$Date.Time = as.POSIXct(bucket$Date.Time)
str(bucket)


#format을 사용해 날짜형식을 변환시킬때는 POSIXct 형식으로 변환되어 있어야 오류가 없음
pis_product[,'Date.Time']=pis_product %>%select(.,Date.Time) %>% format(.,"%Y-%m-%d") 
lims[,'Date.Time']=lims %>%select(.,Date.Time) %>% format(.,"%Y-%m-%d") 
bucket[,'Date.Time']=bucket %>%select(.,Date.Time) %>% format(.,"%Y-%m-%d") 



#데이터 병합(merge방식)
bpa_dataset1 = merge(lims, pis_product,by='Date.Time')  #lims와 pis를 먼저 
bpa_dataset2 = merge(bpa_dataset1,bucket,by='Date.Time')
dim(bpa_dataset2);dim(lims);dim(bucket);dim(pis_product)

sum(is.na(bpa_dataset2))

bpa_dataset=na.omit(bpa_dataset2)


#데이터셋(이상치 제거전) 변수, 데이터 수 체크 
dim(bpa_dataset)


#y값(D50) 이상치 제거하기
colnames(bpa_dataset)
names(bpa_dataset)[2]
names(bpa_dataset)[2]=c("LT0.15")

boxplot(bpa_dataset$D50)
# abline(h=c(1224.411,1659.229),col="red",lty="dotted") 

boxplot(bpa_dataset$D50)$stats
# boxplot.stats(bpa_dataset$D50) 두가지 차이를 실행시켜서 확인해보세요. 



boxplot(bpa_dataset$D50)$stats[1] # Q1-IQR*1.5 
# 이상치 계산 : quantile(bpa_dataset$D50, 1/4) - IQR(bpa_dataset$D50)*1.5
boxplot(bpa_dataset$D50)$stats[5] # Q3+IQR*1.5
# 이상치 계산 : quantile(bpa_dataset$D50, 3/4) + IQR(bpa_dataset$D50)*1.5

bpa_set = bpa_dataset %>% filter(D50>boxplot(D50)$stats[1],D50<boxplot(D50)$stats[5]) %>% select(.,-Date.Time) 

summary(bpa_set)
colnames(bpa_set)
dim(bpa_set)



# 최종 Dataset 저장 
save(bpa_set, file = 'bpa_set')  # 최종 데이터 set 저장
rm(list=ls()) # 모든 데이터 셋 삭제
load('bpa_set') # bpa_set 불러오기 

# write.csv(bpa_set, 'bpa_set.csv')



# 03. 데이터 탐색 -------------------------------------------------------------

#graph분석
pairs(bpa_set)

#boxplot(bpa_set)



# 04. 데이터 Set 구분하기 ------------------------------------------------------

#Data split

dim(bpa_set)
nrow(bpa_set)

set.seed(7279)
train=sample(nrow(bpa_set),nrow(bpa_set)*0.7)
train
test=(1:2582)[-train]
test

length(train)
length(test)


#05. 모형 선정 -----------------------------------------------------------------

#분석실시(modeling) regression, rf

colnames(bpa_set)
lm.fit=lm(D50~.,data=bpa_set[train,-1])
summary(lm.fit)

step(lm.fit)

#최종 회귀식 m(formula = D50 ~ Feed_flow + Feed_Temp + Bucket.RPM_1
        #               + Bucket.RPM_2 + out_AirTemp + Prill.Tower_temp
        #               + Bucke.Hole.Size, data = bpa_set[train,-1])


colnames(bpa_set)
lm.fit_best=lm(D50 ~ Feed_flow + Feed_Temp + Bucket.RPM_1 + Bucket.RPM_2 + 
       + out_AirTemp + Prill.Tower_temp + Bucke.Hole.Size, data = bpa_set[train, -1])

#실제는 다중 공선성 체크하고, 변수 조정해줘야 함


summary(lm.fit_best) # Multiple R-sq 값 체크,  Estimate(부호 체크), *표시 체크 


library(randomForest) ; library(tree)
rf.fit=randomForest(D50~.,data=bpa_set[train,-1],importance=T)
rf.fit
importance(rf.fit)
varImpPlot(rf.fit)



# 06. 모형 선택 --------------------------------------------------------------

# 상관관계 분석 후 제곱해서 계산, 
# but 실질적으로는 회귀 R-sq와, 랜덤포레스트  %var explained를 보고 판단하는 것이맞음

lm_obs = bpa_set[test,]$D50 #실제 관측값
lm_pred = predict(lm.fit_best,newdata=bpa_set[test,-1]) # 예측값 
rf_pred = predict(rf.fit,newdata=bpa_set[test,-1]) # 예측값 

(cor(lm_pred,lm_obs))^2  #상관계수 제곱
(cor(rf_pred,lm_obs))^2  #상관계수 제곱


library(DescTools)
MSE(lm_pred, lm_obs)
MSE(rf_pred, lm_obs)

RMSE(lm_pred, lm_obs) # 회귀식이 상대적으로 우수함
RMSE(rf_pred, lm_obs)

# library(Metrics) # 교육생 문의(6/25)
# bias(lm_pred, lm_obs)

# 07. 모형 최적화 --------------------------------------------------------------

# 모형 선택은 다중선형회귀를 선택함
#최적조건 구하기- 인자별 Range확인 (이상치 제거,  0이하 제거 )

print("Feature range")
for( i in 1:12){
  A = bpa_set %>% filter(bpa_set[,i]>0) %>% .[,i] 
  B = A[A>boxplot(A)$stats[1]&A<boxplot(A)$stats[5]] %>% range()
  print(data.frame(names=colnames(bpa_set)[i] ,lower=B[1],upper=B[2]))
}

bpa_set$Bucke.Hole.Size %>% table()  #Buckek.hole.size는 정수로, 범위 별도 체크 

summary(lm.fit_best) # 부호 다시 체크 

#08. 최적 모형 예측값 ----------------------------------------------------------

# 예측 값을 찾기 위해 회귀식 부호 방향을 체크하고, 그 값을 반영함
# 단, 실무적으로 각 변수의 설정값을 검토 하여 조정 가능함(비용, 안정성 측면 고려)

new=data.frame(Feed_flow=15.2507,         # 음
               Feed_Temp=163.8268,        # 음
               Bucket.RPM_1=83.1646,      # 음
               Bucket.RPM_2=93.2791,      # 음
               out_AirTemp=0.1058,        # 양
               Prill.Tower_temp=56.587,   # 양
               Bucke.Hole.Size=1.3,       # 양
               outAir.flow=806.5549)      # 양 (randomForest를 예측값을 위해 넣음)

new

#최적조건 구하기- 인자별 Range확인 (예측) 
predict(lm.fit_best,newdata=new)
predict(rf.fit,newdata=new)




#참고) 다중공선성 고려했을때 ------------------

colnames(bpa_set)
lm.fit=lm(D50~.,data=bpa_set[train,-1])
summary(lm.fit)

library(car)
vif(lm.fit)

lm.fit = lm(D50 ~ Feed_flow + Feed_Temp + Bucket.RPM_1 + out_AirTemp 
            + Prill.Tower_temp + Bucke.Hole.Size, data = bpa_set[train, -1])

vif(lm.fit)
step(lm.fit)

#최종 회귀식 m(formula = D50 ~ Feed_flow + Feed_Temp + out_AirTemp 
#              + Prill.Tower_temp + Bucke.Hole.Size, data = bpa_set[train,-1])


colnames(bpa_set)
lm.fit_best=lm(D50 ~ Feed_flow + Feed_Temp + out_AirTemp 
                     + Prill.Tower_temp + Bucke.Hole.Size, data = bpa_set[train, -1])

summary(lm.fit_best) # Multiple R-sq 값 체크,  Estimate(부호 체크), *표시 체크 


library(randomForest) ; library(tree)
rf.fit=randomForest(D50~.,data=bpa_set[train,-1],importance=T)
rf.fit
importance(rf.fit)
varImpPlot(rf.fit)



# 최적 모형 선정
lm_obs = bpa_set[test,]$D50 #실제 관측값
lm_pred = predict(lm.fit_best,newdata=bpa_set[test,-1]) # 예측값 
rf_pred = predict(rf.fit,newdata=bpa_set[test,-1]) # 예측값 

(cor(lm_pred,lm_obs))^2  #상관계수 제곱
(cor(rf_pred,lm_obs))^2  #상관계수 제곱


library(DescTools)
MSE(lm_pred, lm_obs)
MSE(rf_pred, lm_obs)

RMSE(lm_pred, lm_obs) # 회귀식이 상대적으로 우수함
RMSE(rf_pred, lm_obs)


# 모형 최종 예측 값 확인 
print("Feature range")
for( i in 1:12){
  A = bpa_set %>% filter(bpa_set[,i]>0) %>% .[,i] 
  B = A[A>boxplot(A)$stats[1]&A<boxplot(A)$stats[5]] %>% range()
  print(data.frame(names=colnames(bpa_set)[i] ,lower=B[1],upper=B[2]))
}

bpa_set$Bucke.Hole.Size %>% table()  #Buckek.hole.size는 정수로, 범위 별도 체크 

summary(lm.fit_best) # 부호 다시 체크 



new=data.frame(Feed_flow=15.2507,         # 음
               Feed_Temp=163.8268,        # 음
               out_AirTemp=0.1058,        # 양
               Prill.Tower_temp=56.587,   # 양
               Bucke.Hole.Size=1.3)       # 양



predict(lm.fit_best,newdata=new)

# ____________________----

# 전지 우수 사례 소개 ----

# DX-LSS MBB과정 --------------------------------------------------------------
# 작성자 : 백인엽P,  최종 수정일 : 20. 7. 6
# Text encoding : UTF-8

# 01. 세부 환경 설정 -----------------------------------------------------------
rm(list=ls())

#set Working directory 
getwd()
setwd("D:/#.Secure Work Folder/DX-LSS-Project/DX-LSS강사과정/2020강사과정/전지")
dir()


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


#data 현수준 파악을 위한 데이터 저장
write.csv(A, 'A_Model.csv')


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
test= (1:270)[-train]



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
