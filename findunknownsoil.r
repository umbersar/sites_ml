
setwd("C:/Users/horat/Desktop/CSIROIntership/soilCode")

library(dplyr)

#create pivot table 
library(reshape)
library(data.table)

#data partition seperate trainset and testset
library (caTools)

library(caret)

#svm library due to limitation of iterations change the library
library(e1071)
library(LiblineaR)

#random forest
library(randomForest)

#ID4 Decision Tree classifier(CART)
library(rpart)
library(rpart.plot)
library(rattle)

#xgboost
library(xgboost)

#for knn classification
library(class)

#install neuralnetwork
library(neuralnet)

#adabag library
library(adabag)

#Stochastic Gradient Descent (SGD) Method Learning Function
library(gradDescent)
library(lightgbm)
#https://www.kaggle.com/c/amazon-employee-access-challenge/discussion/5128#38925

#matrix library
library(Matrix)

#catboost
library(catboost)

#fast naive bayes
library("fastNaiveBayes")

#tidyverse for easy data manipulation and visualization
#caret for easy machine learning workflow

library(tidyverse)
library(caret)

#mlp
library(RSNNS)

featureSoilTable <- read.csv(file = "featureTable.csv",stringsAsFactors=FALSE)
#normalize the labr_value name
#preprocessing the data
normalize <-function(y) {
  
  x<-y[!is.na(y)]
  
  x<-(x - min(x)) / (max(x) - min(x))
  
  y[!is.na(y)]<-x
  
  return(y)
}

#change the NULL to na
featureSoilTable['h_texture'][featureSoilTable['h_texture'] == "NULL"] <- NA
#add appendix to colname:
colnames(featureSoilTable) <- paste("Str",colnames(featureSoilTable),sep = "_")

print(featureSoilTable)

#extract valid and invalid soil sample
validsoilTexture <- featureSoilTable[!is.na(featureSoilTable$Str_h_texture),]
invalidsoilTexture <- featureSoilTable[is.na(featureSoilTable$Str_h_texture),]
validsoilTexture <- validsoilTexture[,colSums(is.na(validsoilTexture))<nrow(validsoilTexture)]
#remove rows have less than 4 data
contribution <- as.data.frame(rowsum(rep(1,times = length(validsoilTexture$Str_h_texture)), validsoilTexture$Str_h_texture),row.names = count)
label <- sort(unique(validsoilTexture$Str_h_texture))
contribution <- cbind(label,contribution)
invaliddata <- contribution[contribution$V1 < 4,]

validsoilTexture <- cbind(Isknown = 1, validsoilTexture)

for (l in invaliddata$label){
  rowlist = which(validsoilTexture$Str_h_texture == l)
  #print(rowlist)
  validsoilTexture[rowlist,]$Isknown = 0
}

validsoilTexture <- validsoilTexture[,-c(2)]

validsoilTexture$Isknown <- as.numeric((validsoilTexture$Isknown))
validsoilTexture[,-c(1)] <- apply(apply(validsoilTexture[,-c(1)], 2, as.factor), 2, as.numeric)

validsoilTexture[,-c(1)] <- (apply(validsoilTexture[,-c(1)],2,normalize))
validsoilTexture <- as.data.frame(validsoilTexture)

#change null value to 0
validsoilTexture[is.na(validsoilTexture)] = 0

print(validsoilTexture)
ncol <- ncol(validsoilTexture)

#set random seed
set.seed(122)

#give the valid sample
split = sample.split(validsoilTexture$Isknown,SplitRatio = 0.7)

train_set = subset(validsoilTexture, split == TRUE)
test_set = subset(validsoilTexture, split == FALSE)

model <- naiveBayes(as.factor(Isknown) ~ .,data = train_set,laplace =2)
pred <- predict(model,train_set[,-1])
table(pred, train_set$Isknown)

svmClassifier <- LiblineaR(data = train_set[,-1],target = train_set[,c("Isknown")],bias=1,cost = 1000)
svmPredictTrain <- predict(svmClassifier,train_set[,-1],proba=TRUE,decisionValues=TRUE)
svmPredictTrainTable <- table(svmPredictTrain$predictions,train_set[,c("Isknown")])

svmPredictTest <- predict(svmClassifier,test_set[,-1],proba=TRUE,decisionValues=TRUE)
svmPredictTestTable <- table(svmPredictTest$predictions,test_set[,c("Isknown")])

