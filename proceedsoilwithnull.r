
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

featureSoilTable <- read.csv(file = "featureTable.csv")

#normalize the labr_value name
#preprocessing the data
normalize <- function(y){
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

#remove all columns with na
validsoilTexture <- validsoilTexture[,colSums(is.na(validsoilTexture))<nrow(validsoilTexture)]

validsoilTexture$Str_h_texture <- as.numeric(as.factor(validsoilTexture$Str_h_texture))
validsoilTexture <- apply(validsoilTexture, 2, as.factor)
validsoilTexture <- apply(validsoilTexture, 2, as.numeric)
validsoilTexture <- as.data.frame(apply(validsoilTexture,2,normalize))

countype <- length(unique(validsoilTexture$Str_h_texture))
validsoilTexture$Str_h_texture <- floor(validsoilTexture$Str_h_texture * countype)
ncol <- ncol(validsoilTexture)

#set random seed
set.seed(122)

#give the valid sample
split = sample.split(validsoilTexture$Str_h_texture,SplitRatio = 0.7)

train_set = subset(validsoilTexture, split == TRUE)
test_set = subset(validsoilTexture, split == FALSE)

summary(train_set)
#train_set.scale = scale(train_set.num,center=  TRUE,scale = TRUE)

#lightGBM
train_set.num_X <- select (train_set,-c(Str_h_texture))
test_set.num_X <- select (test_set,-c(Str_h_texture))

ltrain = lgb.Dataset(data = as.matrix(train_set.num_X),label = train_set$Str_h_texture, free_raw_data = FALSE)
params <- list(objective="regression", metric="l2")
model <- lgb.cv(
  params, 
  ltrain ,
  10,
  nfold=5,
  min_data=1, 
  learning_rate=1, 
  early_stopping_rounds=10,
  Depth = 1,
  lambda_l1 = 6,
  lambda_l2 = 10,
  num_leaves = 2,
  num_iterations = 1000,
  min_gain_to_split = 500,)

ltest = lgb.Dataset.create.valid(ltrain , as.matrix(test_set.num_X), label = test_set$Str_h_texture)
valids <- list(test = ltest)

grid_search <- expand.grid(Depth = 1:8,
                           L1 = 0:10,
                           L2 = 5:10)

model <- list()
perf <- numeric(nrow(grid_search))

for (i in 1:nrow(grid_search)) {
  model[[i]] <- lgb.train(list(objective = "regression",
                               metric = "l2",
                               lambda_l1 = grid_search[i, "L1"],
                               lambda_l2 = grid_search[i, "L2"],
                               max_depth = grid_search[i, "Depth"]),
                          ltrain,
                          2,
                          valids,
                          min_data = 1,
                          learning_rate = 1,
                          early_stopping_rounds = 5,
                          num_leaves = 2,
                          num_iterations = 1000,
                          min_gain_to_split = 500,)
  
  perf[i] <- min(rbindlist(model[[i]]$record_evals$test$l2))
}

cat("Model ", which.min(perf), " is lowest loss: ", min(perf), sep = "")

print(grid_search[which.min(perf), ])

