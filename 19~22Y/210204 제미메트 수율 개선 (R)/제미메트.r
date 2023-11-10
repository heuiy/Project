# ___________________________--------
# **************************************--------
# ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁====
# 02 _ 210204 제미메트 수율 개선 ----
# ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁ _ ◁====
# ___________________________--------  
# **************************************--------

# Package Loading -----

pkg <- c("car", "glmnet", "caret", "MASS", "randomForest", "dplyr", "DescTools", "reshape2", "ISLR", "readxl", "tidyr", "skimr")

sapply(pkg, require, character.only=T)

# Data Loading -----

zemi <- read_excel("제미메트 서방정 25_1000.xlsx")

str(zemi)
dim(zemi)
is.na(zemi)

skim(zemi)
head(zemi)

# 결측치로 구성된 열 / 결측치가 50% 이상인 열 제거

zemi_na <- zemi[, c(-4:-6, -14:-16, -30:-31, -33:-35, -37:-39, -41:-43, -45:-46, -48:-49)]
skim(zemi_na)
zemi_na <- zemi_na[, c(-23, -30, -31, -33, -34, -36, -37, -39, -40, -42. -43, -45, -46, -48, -49)]
zemi_na <- zemi_na[, c(-33, -35, -40:-46, -50)]
skim(zemi_na)
zemi_na <- zemi_na[, c(-33, -44, -57, -58, -63, -64)]
skim(zemi_na)

# 형 변환 char --> numeric

# 문자열 분리

class(zemi_na$`exhale temp`)

#zemi_na$`exhale temp`<- zemi_na$`exhale temp` %>% strsplit("~") 

exhale <- data.frame(do.call("rbind",
                            strsplit(as.character(zemi_na$`exhale temp`), split = "~", fixed = T)))  
colnames(exhale)[1:2] <- c("exhale_low", "exhale_high")
exhale$exhale_low <- as.numeric(exhale$exhale_low); exhale$exhale_high <- as.numeric(exhale$exhale_high)

refine <- data.frame(do.call("rbind",
                             strsplit(as.character(zemi_na$`refine temp`), split = "~", fixed = T)))
colnames(refine)[1:2]=c("refine_low", "refine_high")
refine$refine_low <- as.numeric(refine$refine_low); refine$refine_high <- as.numeric(refine$refine_high)

zemi_split <- cbind(zemi_na, exhale)
zemi_split <- cbind(zemi_split, refine)

zemi_split <- zemi_split[, c(-2, -53, -54)] # 문자열 분리 전 원래 colname 제거

# 결측치를 각 열의 평균값으로 대체

skim(zemi_split)

zemi_sub <- data.frame(sapply(zemi_split[, -1], function(x) ifelse(is.na(x), mean(x, na.rm = T), x)))
# 위의 구문 실행 시 data.frame 중 chr 이 있으면 전체가 다 chr 로 변하는 현상??

write.csv(zemi_sub, file = "zemi_sub", row.names = T)
skim(zemi_sub)
str(zemi_sub)

colnames(zemi_sub)[56] <- c("Y")

# Dataset 분리

set.seed(1234)

sn <- sample(1:nrow(zemi_sub), size = nrow(zemi_sub)*0.8)

train <- zemi_sub[sn, ]
test <- zemi_sub[-sn, ]

str(train)
str(test)

# Multiple Linear Model (일반적 Case, CV 검증)-----

#idx <- createFolds(zemi_sub$Y, k = 4) # dataset 이 작아서 교차 검증(4개)으로 훈련/시험 data 분류
#train <- data.frame(zemi_sub[-idx$Fold4, ])
#test <- data.frame(zemi_sub[idx$Fold4, ])

lm.model <- lm(Y ~ ., data = train) # 선형 모델 적합
lm.step <- stepAIC(lm.model, direction = "both") # stepwise
summary(lm.step) # stepwise 결과 확인
plot(lm.step$anova$AIC) # stepwise 결과 AIC 가 얼마나 좋아지는지 확인

lm.model_best <- lm(Y ~ filter.impeller.speed1 + exhale.speed1 + temp.on.spay1 + 
                      loss.on.drying2 + met.rash.weight + met.rash.yield + jemi.rash.weight + 
                      avg.Pressure1 + avg.Pressure5 + charging.depth1 + charging.depth5 + 
                      down.punch.depth4 + down.punch.depth7 + up.punch.depth4 + 
                      down.punch.depth10 + Suctin1 + avg.weight1 + avg.thickness1 + 
                      avg.thickness3 + avg.friability3 + rash.weight2 + loss.total.weight + 
                      containing.weight + humidity + spray.height + containg.weight + 
                      loss.tablet.weight + total.weight + gap.od.tablet.and.sorting + 
                      exhale_low + exhale_high, data = train)
summary(lm.model_best)

lm_obs <- test$Y
lm_pred <- predict(lm.model_best, newdata = test)

MSE(lm_obs, lm_pred) # 모형 성능 확인(MSE)
RMSE(lm_obs, lm_pred) # 모형 성능 확인 (RMSE)
VIF(lm.model_best) # 추정된 회귀 계수의 다중 공선성 확인

plot(lm.model_best$coefficients, ylim = c(-10, 10)) # 회귀계수들을 Plot

# Multiple Linear Model (glmnet package 이용) -----

set.seed(1234)

x <- model.matrix(zemi_sub$filter.impeller.speed1 ~ . , data = zemi_sub)[, -1] # 독립변수 matrix
y <- zemi_sub$Y # 종속변수

train_ridge <- sample(1:nrow(x), size = nrow(x)*0.8)

test_ridge <- x[-train_ridge, ]
y.test <- y[-train_ridge]

str(test_ridge)

# Multiple Linear Model 생성

sh <- 10^seq(10, -2, length = 1000)
ridge_tr <- glmnet(x[train_ridge,], y[train_ridge], alpha = 0, lambda = sh)
multi.lm_ridge <- lm(y ~ x, subset = train_ridge) # 다중 선형 모델 적합
summary(multi.lm_ridge)

multi.lm_coef <- predict(ridge_tr, s = 0, newx = test_ridge, type = "coef") # 다중 선형 모델 회귀 계수 확인
multi.lm_pred <- predict(ridge_tr, s = 0, newx = test_ridge) 

MSE(y.test, multi.lm_pred) # 다중 선형 모델 MSE 
plot(multi.lm_coef, ylim = c(-10, 10))
VIF(multi.lm_ridge) # 독립 변수 간에 완전공선성이 존재하므로 계수 추정 불가능
# 위의 식에서 모든 λ가 0인 경우, 모든 독립변수 x는 선형독립이다. 
# 반면 하나라도 0이 아닌 λ가 있다면 선형종속이라고 한다. 선형종속인 경우를 완전 다중공선성(perfect multicollinearity)이라고 한다.

# Ridge Regression Model 분석 실시 -----

set.seed(1234)

x <- model.matrix(zemi_sub$filter.impeller.speed1 ~ . , data = zemi_sub)[, -1] # 독립변수 matrix
y <- zemi_sub$Y # 종속변수

# Data 분리

train_ridge <- sample(1:nrow(x), size = nrow(x)*0.8)

test_ridge <- x[-train_ridge, ]
y.test <- y[-train_ridge]
  
str(test_ridge)

# Ridge Regression Model 생성
sh <- 10^seq(10, -2, length = 1000)
ridge_tr <- glmnet(x[train_ridge,], y[train_ridge], alpha = 0, lambda = sh)
dim(coef(ridge_tr))
plot(ridge_tr)
summary(ridge_tr)

# Ridge Regression Model 검증
set.seed(1234)

cv_ridge <- cv.glmnet(x[train_ridge, ], y[train_ridge], alpha = 0) # 교차 검증을 통한 람다값 추정
plot(cv_ridge)

bestlamdba_ridge <- cv_ridge$lambda.min
ridge_tr_best <- glmnet(x[train_ridge,], y[train_ridge], alpha = 0, lambda = cv$lambda.min)
ridge_pred <- predict(ridge_tr, s = bestlamdba, newx = test_ridge)

MSE(y.test, ridge_pred)
RMSE(y.test, ridge_pred)

vif(ridge_tr_best)

plot(predict(ridge_tr_best, newx = x[test,], type = "coef"), ylim = c(-10,10))

# Lasso Regression Model 생성

lasso_tr <- glmnet(x[train_ridge,], y[train_ridge], alpha = 1, lambda = sh)
plot(lasso_tr)

set.seed(1234)

cv_lasso <- cv.glmnet(x[train_ridge, ], y[train_ridge], alpha = 1) # 교차 검증을 통한 람다값 추정
plot(cv_lasso)

bestlamdba_lasso <- cv_lasso$lambda.min
lasso_tr_best <- glmnet(x[train_ridge,], y[train_ridge], alpha = 1, lambda = cv_lasso$lambda.min)
lasso_pred <- predict(lasso_tr, s = bestlamdba_lasso, newx = test_ridge) 

MSE(y.test, lasso_pred)
RMSE(y.test, lasso_pred)

vif(lasso_tr_best)

plot(predict(lasso_tr_best, newx = x[test,], type = "coef"), ylim = c(-10,10))

#_______________________----

install.packages("devtools") # devtools 패키지
library("devtools")

devtools::install_github("hogi76/dxlss", force=TRUE) # LG화학 DX-LSS 패키지 설치 
library(dxlss)

library("dxlss") # Tip. R Studio 신규 구동 시마다 실행하세요. 
pre_check() # Tip.  R Studio 신규 구동 시마다 실행하세요. 


# Measure 단계  ----------------------------------------------------------------

# 데이터 전처리(project Y)

zemi <- read.csv("PSS2_LG.csv")  # Data 불러오기
zemi  # 제미메트 공정 Data
head(zemi) # Data 확인
?prestep1 # prestep1 - 데이터 확인 함수 사용방법 확인
prestep1(zemi, zemi$total.weight) # pss1 데이터 확인

?prestep2 # prestep2 - 전처리 함수 사용방법 확인
head(prestep2(zemi, zemi$total.weight)) # zemi 데이터 전처리(이상치, 결측치 제거)

zemi1.1 = prestep2(zemi, zemi$total.weight) # 전처리된 데이터 zemi1.1 에 담음

prestep1(zemi1.1, zemi1.1$total.weight) # 전처리 데이터인 zemi1.1 데이터 최종확인

# 공정능력 산출

?step1 # 공정능력 분석 함수 사용방법 확인 
# step1 함수 사용은 xST는 단기 데이터,  LSL = 규격 하한값, USL = 규격상한값, Target = 목표값에 대해서 입력 후 실행

# total.weight 에 대한 공정능력분석
step1(xST = zemi1.1$total.weight, xLT = NA, LSL = 430, USL = 460, Target = 450)


# Analyze 단계  ----------------------------------------------------------------

?prestep3 # prestep3 함수 사용방법 확인

# prestep3함수를 통해 훈련용 데이터셋과 검증용 데이터셋으로 나눔
head(prestep3(zemi1.1)$train) # 훈련용 데이터셋 할당
head(prestep3(zemi1.1)$test)  # 검증용 데이터셋 할당
zemi2.2 = prestep3(zemi1.1)$train # 훈련용 데이터 셋 
zemi2.3 = prestep3(zemi1.1)$test # 검증용 데이터 셋

# 회귀모형과 랜덤포레스트을 이용한 분석
step2(zemi2.2, zemi2.3, zemi2.2$total.weight~., zemi2.3$total.weight~., zemi2.3$total.weight)


# Improve 단계  ----------------------------------------------------------------

# 인자의 범위 확인 함수
?step3() # step3 함수 사용방법 확인
step3(zemi1.1)  # zemi1.1의 각 인자 범위 확인 함수 실행


# project Y 예측을 위한 변수 값 선정 및 데이터 프레임 

newx = data.frame(filter.impeller.speed1 = 500, exhale.speed1 = 3000, avg.Pressure1 = 3.4, avg.Pressure5 = 35.08, 
                  down.punch.depth7 = 16.83, up.punch.depth4 = 2.52, avg.thickness1 = 8.30, avg.thickness2 = 8.30,
                  avg.thickness3 = 8.13, avg.hardness1 = 16.3, avg.friability1 = 0.03, total.weight = 449.75)
# 예측하고자 하는 변수의 값을 입력하여 데이터 프레임으로 만듬 / project Y의 값의 특성에 따라 변수의 선택은 달라지며, 
# project Y의 특성이 망대이면, 회귀계수 값의 부호 방향이 양수 인 경우에는 큰 값을 선택하고, 음수이면 작은 값을 선택함. 
# project Y의 특성이 망소이면, 회귀계수 값의 부호 방향이 양수 인 경우에는 작은 값을 선택하고, 음수이면 큰 값을 선택함.
# 회귀 계수 값의 p-value가 0.05보다 크고, 유의하지 않으면 평균값을 취함

# project Y 예측값 산출 

?step4  # 선택된 변수들의 데이터 프레임인 newx의 데이터프레임 기반으로 기존에 도출한 회귀식을 활용하여, project Y의 예측값을 도출함
step4(zemi1.1, newx, zemi1.1$total.weight~., newx$total.weight~., newx$total.weight)


# Control 단계  ----------------------------------------------------------------

?poststep1   # I-MR 관리도 사용방법 확인
poststep1(zemi1.1$total.weight) # total.weight 데이터를 I-MR 관리도 실행
