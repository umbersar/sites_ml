
setwd("C:/Users/horat/Desktop/CSIROIntership/soilCode")

#data partition seperate trainset and testset
library (caTools)

library(caret)
#svm library
library(e1071)

#random forest
library(randomForest)

#ID4 Decision Tree classifier(CART)
library(rpart)
library(rpart.plot)
library(rattle)

#load the soil data
soil <- read.csv(file = "hr_lr_labm.csv")

#for knn classification
library(class)

#install neuralnetwork
library(neuralnet)
#reduce the amount to 10000
soil <- head(soil,n=1000L)
rSoil <- nrow(soil)



#eliminate duplicate row
Predata <- soil[1:7]
Predata <- Predata[!duplicated(Predata[1:7]),]
rPre <- nrow(Predata)

featurerow <- soil$labm_code
featurerow <- featurerow[!duplicated(featurerow)]
featureTable <- data.frame(matrix(ncol=length(featurerow),nrow=rPre, dimnames=list(NULL, featurerow)))

Predata <- cbind(Predata,featureTable)

for (i in (1:rSoil)){
  for (j in (1:rPre)){
    if(identical(soil[1:7][i,],Predata[1:7][j,])){
      labcode = soil["labm_code"][i,]
      labvalue = soil["labr_value"][i,]
      Predata[labcode][j,] = labvalue
    }
  }
}
  

#combine two dataframes
#soil <- cbind(soil,featureTable)


#factorize the soil
# labmNum <- factor(c(labmCode))
# 
# soil$labm_code <- as.numeric(labmNum)
# 
# #change labr_value to numeric
# soil$labr_value <- as.numeric(soil$labr_value)
# 
# #load the "h_soil_water_stat" non-null column
# #fill in the NULL value as NA value
# soil$h_soil_water_stat[soil$h_soil_water_stat == "NULL"] = NA
# 
# #get the null value row
# naValuerow <- which(is.na(soil$h_soil_water_stat))
# 
# validsoilSample <- soil[-naValuerow,]
# 
# invalidsoilSample <- soil[naValuerow,]
# 
# soil_label = validsoilSample$h_soil_water_stat
# soil_label <- factor(c(soil_label))
# validsoilSample$h_soil_water_stat <- as.numeric(soil_label)
# 
# #set random seed
# set.seed(122)
# 
# #give the valid sample
# validsoilSample <- validsoilSample[,c("h_soil_water_stat","labm_code","labr_value")]
# 
# 
# split = sample.split(validsoilSample$h_soil_water_stat,SplitRatio = 0.75)
# train_set = subset(validsoilSample, split == TRUE)
# test_set = subset(validsoilSample, split ==FALSE)
# 
# # x <- subset(validsoilSample, select=-h_soil_water_stat)
# # y <- validsoilSample$h_soil_water_stat
# 
# #to check if which value is the best
# # svm_tune <- tune(svm, train.x=x, train.y=y, 
# #                  kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
# # 
# # print(svm_tune)
# 
# 
# #svm classifier
# svmClassifier = svm(formula = h_soil_water_stat ~ .,
#                     data = train_set,
#                     type="C-classification",
#                     kernel = 'radial',
#                     cost = 10,
#                     gamma = 0.01)
# 
# tuned <- tune.svm(h_soil_water_stat ~.,
#                   data = train_set,
#                   gamma = 10^(-1:-3), cost = 10^(1:3), 
#                   tunecontrol = tune.control(cross = 10))
# 
# summary(tuned)
# train_set$h_soil_water_stat <- as.character(train_set$h_soil_water_stat)
# train_set$h_soil_water_stat <- as.factor(train_set$h_soil_water_stat)
# test_set$h_soil_water_stat <- as.character(test_set$h_soil_water_stat)
# test_set$h_soil_water_stat <- as.factor(test_set$h_soil_water_stat)
# # Predicting the Test set results 
# y_Svm_test_pred <- predict(svmClassifier,newdata = test_set[,c("labm_code","labr_value")])
# y_Svm_train_pred <- predict(svmClassifier,newdata = train_set[,c("labm_code","labr_value")])
# 
# cm_SVMTest= table(y_Svm_test_pred,test_set[,c("h_soil_water_stat")])
# cm_SVMTrain = table(train_set[,c("h_soil_water_stat")],y_Svm_train_pred)
# 
# #compute the score
# cm_SVMTestScore <- (sum(diag(cm_SVMTest)))/sum(cm_SVMTest)
# cm_SVMTrainScore <-  (sum(diag(cm_SVMTrain)))/sum(cm_SVMTrain)
# #randomForest Classifier
# RfClassifier = randomForest(h_soil_water_stat ~ .,data = train_set,ntree = 10,proximity = T)
# 
# rfTable <- table(predict(RfClassifier),train_set$h_soil_water_stat)
# print(RfClassifier)
# plot(RfClassifier)
# 
# #Classification with CART model
# cartFit <- rpart(h_soil_water_stat ~ .,data = train_set,control = rpart.control(cp = 0.0001))
# #get cp value
# printcp(cartFit)
# #we can prune data with the CP value that contains the lowest error.
# fit.pruned = prune(cartFit, cp = 0.00012488)
# cartPrediction <- predict(fit.pruned, test_set, type = "class")
# data.frame(test_set,cartPrediction)
# confusionMatrix(test_set$h_soil_water_stat,cartPrediction)
# 
# #classification with KNN model
# #knnClassifer <- knn(train_set,test_set)
# 
# 
# #NaiveBayes classification
# # The formula is traditional Y~X1+X2+бн+Xn
# # The data is typically a dataframe of numeric or factor variables.
# # laplace provides a smoothing effect (as discussed below)
# # subset lets you use only a selection subset of your data based on some boolean filter
# # na.action lets you determine what to do when you hit a missing value in your dataset.
# nbClassifier <- naiveBayes(h_soil_water_stat ~ .,data = train_set,laplace=1)
# nbTestPrediction <- predict(nbClassifier,test_set,type = "class")
# nbTableTest <- table(nbTestPrediction,test_set$h_soil_water_stat)
# nbtestTable <- (sum(diag(nbTableTest)))/sum(nbTableTest)
# 
# nbTrainPrediction <- predict(nbClassifier,train_set,type = "class")
# nbTableTrain <- table(nbTrainPrediction,train_set$h_soil_water_stat)
# nbtrainTable <- (sum(diag(nbTableTrain)))/sum(nbTableTest)
# 
# #neuro network
# #Min-Max normalization
# Nortrain <- train_set
# normalize <- function(x) {
#   return ((x - min(x)) / (max(x) - min(x)))
# }
# Nortrain$h_soil_water_stat <- normalize(as.numeric(train_set$h_soil_water_stat))
# Nortrain$labm_code <- normalize(as.numeric(train_set$labm_code))
# Nortrain$labr_value <- normalize((as.numeric(train_set$labr_value)))
# 
# Nortest <- test_set
# Nortest$h_soil_water_stat <- normalize(as.numeric(test_set$h_soil_water_stat))
# Nortest$labm_code <- normalize(as.numeric(test_set$labm_code))
# Nortest$labr_value <- normalize((as.numeric(test_set$labr_value)))
# 
# set.seed(222)
# nnClassifier <- neuralnet(h_soil_water_stat ~ .,data=Nortrain, hidden=4,linear.output = F)
# print(nnClassifier$result.matrix)
# plot(nnClassifier)
# 
# #prediction
# output<- compute(nnClassifier,Nortrain[,-1])
# 
# #confusion Matrix & Misclassifcation Error - training data