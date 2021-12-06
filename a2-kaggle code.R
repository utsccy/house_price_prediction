#devtools::install_github("ropenscilabs/skimr")
if (!require("randomForest")) install.packages("randomForest")
if (!require("skimr")) install.packages("skimr")
library(tidyverse)
library(skimr)
library(mice)
library(glmnet)
library(e1071)
library(caret)
library(car)
library(GGally)

options(scipen=999)

#Load Dataset

setwd("~/kaggle_houseprice")
train<-read.csv('train.csv',stringsAsFactors = FALSE)
test<-read.csv('test.csv',stringsAsFactors = FALSE)
test_id <- test$Id
train<-select(train,-'Id')



#Outlier Analysis

boxplot(train$SalePrice)
plot(train$GrLivArea,train$SalePrice)
train[which(train$GrLivArea>4000&train$SalePrice<350000),]
train<-train[-which(train$GrLivArea>4000&train$SalePrice<350000),]


train_x<-select(train,-'SalePrice')
train_y<-select(train,'SalePrice')
test_x<-select(test,-'Id')

#histogram
hist(train_y$SalePrice)
y<-log(train_y+1)
y = y[[1]]

#combine train and test to one dataset

full<-rbind(train_x,test_x)

drop<-c('Utilities')
full<-full[,!(names(full)%in%drop)]


###########################
####Feature Engineering####
###########################
full$YrSold <- as.character(full$YrSold)
full$MoSold <- as.character(full$MoSold)
full$MSSubClass <- as.character(full$MSSubClass)
full$IsNew <- as.character(ifelse(full$YrSold<=1+full$YearBuilt, 1, 0))


####################
####MISSING DATA####
####################

which(colSums(is.na(full)) > 0)

#Garage

full$GarageCars[is.na(full$GarageCars)]<-0
full$GarageYrBlt[is.na(full$GarageYrBlt)]<-0
full$GarageArea[is.na(full$GarageArea)]<-0
full$GarageType[is.na(full$GarageType)]<-'None'
full$GarageFinish[is.na(full$GarageFinish)]<-'None'
full$GarageQual[is.na(full$GarageQual)]<-'None'
full$GarageCond[is.na(full$GarageCond)]<-'None'

#Basement

full$BsmtFinSF1[is.na(full$BsmtFinSF1)]<-0
full$BsmtFinSF2[is.na(full$BsmtFinSF2)]<-0
full$BsmtUnfSF[is.na(full$BsmtUnfSF)]<-0
full$TotalBsmtSF[is.na(full$TotalBsmtSF)]<-0
full$BsmtFullBath[is.na(full$BsmtFullBath)]<-0
full$BsmtHalfBath[is.na(full$BsmtHalfBath)]<-0
full$BsmtQual[is.na(full$BsmtQual)]<-'None'
full$BsmtExposure[is.na(full$BsmtExposure)]<-'None'
full$BsmtFinType1[is.na(full$BsmtFinType1)]<-'None'
full$BsmtFinType2[is.na(full$BsmtFinType2)]<-'None'
full$BsmtCond[is.na(full$BsmtCond)]<-'None'

#other
full$MasVnrArea[is.na(full$MasVnrArea)]<-0
full$MasVnrType[is.na(full$MasVnrType)]<-'None'
full$Alley[is.na(full$Alley)]<-'None'
full$PoolQC[is.na(full$PoolQC)]<-'None'
full$MiscFeature[is.na(full$MiscFeature)]<-'None'
full$Fence[is.na(full$Fence)]<-'None'
full$FireplaceQu[is.na(full$FireplaceQu)]<-'None'
full$MSZoning[is.na(full$MSZoning)]<-'None'
full$KitchenQual[is.na(full$KitchenQual)]<-'None'
full$Functional[is.na(full$Functional)]<-'None'
full$Electrical[is.na(full$Electrical)]<-'None'
full$Exterior1st[is.na(full$Exterior1st)]<-'None'
full$Exterior2nd[is.na(full$Exterior2nd)]<-'None'
full$SaleType[is.na(full$SaleType)]<-'None'
full$LotFrontage[is.na(full$LotFrontage)]<-median(full$LotFrontage,na.rm = TRUE)



#correlation

ggcorr(full)
full <- select(full, -c(TotRmsAbvGrd, GarageArea,X2ndFlrSF, X1stFlrSF))

#devide variables into character and integer


chr_var <- full[,sapply(full,is.character)==TRUE]
int_var <- full[,sapply(full,is.character)==FALSE]

#boxcox transformation

int_skew <- sapply(names(int_var), function(x) {
  skewness(full[[x]], na.rm = TRUE)
})

int_skew <- int_skew[abs(int_skew) > 0.5]

for (x in names(int_skew)) {
  bc = BoxCoxTrans(full[[x]], lambda = 0.1)
  int_var[[x]] = predict(bc, full[[x]])
}

#normalize the data
preint <- preProcess(int_var, method=c("center", "scale"))
int_var <- predict(preint, int_var)

#one-hot encoding - dummy variables

dummies <- dummyVars(~., full[names(chr_var)])
chr_var <- predict(dummies, full[names(chr_var)])


#combine to full dataset
full<-cbind(chr_var,int_var)
#skim(full)


#model!
train <- full[1:nrow(train_x),]
test<-full[(nrow(train_x)+1):nrow(full),]


x = as.matrix(train)
x_test=as.matrix(test)

cv_lasso = cv.glmnet(x, y,alpha=1)
penalty.lasso <- cv.lasso$lambda.min
penalty.lasso
lasso.opt.fit <-glmnet(x = x, y = y, alpha = 1, lambda = penalty.lasso)
lasso.train <- predict(lasso.opt.fit, s = penalty.lasso, newx =x)
sqrt(mean((y - lasso.train)^2))
lasso_test <- predict(cv_lasso, newx = x_test, s = "lambda.min")
lasso.final<-cbind(test_id,(exp(lasso_test)-1))
colnames(lasso.final)<-c('Id','SalePrice')
write.csv(lasso.final, '../kaggle_houseprice/submission_lasso.csv',row.names=FALSE)

#ridge
cv.ridge = cv.glmnet(x, y, alpha = 0)
plot(cv.ridge)
penalty.ridge <- cv.ridge$lambda.min #determine optimal penalty parameter, lambda
penalty.ridge #see where it was on the graph
# plot(crossval,xlim=c(-8.5,-6),ylim=c(0.006,0.008)) # lets zoom-in
ridge.opt.fit <-glmnet(x = x, y = y, alpha = 0, lambda = penalty.ridge) #estimate the model with the optimal penalty
coef(ridge.opt.fit) #resultant model coefficients
ridge.train <- predict(ridge.opt.fit, s = penalty.ridge, newx =x)
sqrt(mean((y - ridge.train)^2))
ridge.testing <- predict(ridge.opt.fit, s = penalty.ridge, newx =x_test)
ridge.final<-cbind(test_id,(exp(ridge.testing)-1))
colnames(ridge.final)<-c('Id','SalePrice')
write.csv(ridge.final, '../kaggle_houseprice/submission_2_ridge.csv',row.names=FALSE)