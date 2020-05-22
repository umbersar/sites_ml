
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

#extract valid and invalid soil sample
validsoilTexture <- featureSoilTable[!is.na(featureSoilTable$Str_h_texture),]
invalidsoilTexture <- featureSoilTable[is.na(featureSoilTable$Str_h_texture),]

# remove columns that only have nas
validsoilTexture <- validsoilTexture[,colSums(is.na(validsoilTexture))<nrow(validsoilTexture)]

#remove rows have less than 4 data
contribution <- as.data.frame(rowsum(rep(1,times = length(validsoilTexture$Str_h_texture)), validsoilTexture$Str_h_texture),row.names = count)
label <- sort(unique(validsoilTexture$Str_h_texture))
contribution <- cbind(label,contribution)
invaliddata <- contribution[contribution$V1 < 4,]

for (l in invaliddata$label){
  rowlist = which(validsoilTexture$Str_h_texture == l)
  #print(rowlist)
  validsoilTexture <- validsoilTexture[-rowlist,]

}

validsoilTexture$Str_h_texture <- as.numeric(as.factor(validsoilTexture$Str_h_texture))
validsoilTexture[,-1] <- apply(apply(validsoilTexture[,-1], 2, as.factor), 2, as.numeric)
validsoilTexture[,-1]<- (apply(validsoilTexture[,-1],2,normalize))
validsoilTexture <- as.data.frame(validsoilTexture)

#change null value to 0
validsoilTexture[is.na(validsoilTexture)] = 0

print(validsoilTexture)
ncol <- ncol(validsoilTexture)

#set random seed
set.seed(122)

#give the valid sample
split = sample.split(validsoilTexture$Str_h_texture,SplitRatio = 0.7)

train_set = subset(validsoilTexture, split == TRUE)
test_set = subset(validsoilTexture, split == FALSE)

train_set$Str_h_texture = as.numeric(train_set$Str_h_texture)
test_set$Str_h_texture = as.numeric(test_set$Str_h_texture)

train_set.num_X <- select (train_set,-c(Str_h_texture))
test_set.num_X <- select (test_set,-c(Str_h_texture))

print(featureSoilTable)
#print each element contribution

summary(train_set)
#train_set.scale = scale(train_set.num,center=  TRUE,scale = TRUE)


# Find the best model with the best cost parameter via 10-fold cross-validations

# the tunning part of svm, which will take lots of time to run

tryTypes=c(0:7)
tryCosts=c(1000,1,0.001)
bestCost=NA
bestAcc=0.2
bestType=NA

for(ty in tryTypes){

   for(co in tryCosts){
    acc=LiblineaR(data=train_set[,-1],target=train_set[,c("Str_h_texture")],type=7,cost=co,bias=1,verbose=FALSE)
    cat("Results for C=",co," : ",acc," accuracy.\n",sep="")
    if(acc>bestAcc){
    bestCost=co
    bestAcc=acc
    bestType=ty
    }
  }

}

# svm classifier The train score of svm algorithm is  0.32799 
# The test score of svm algorithm is  0.3019961
# the running time of svm is approximately 40.66578 seconds

svmClassifier <- LiblineaR(data = train_set[,-1],target = train_set[,c("Str_h_texture")],bias=1,cost = 1000)
svmPredictTrain <- predict(svmClassifier,train_set[,-1],proba=TRUE,decisionValues=TRUE)
svmPredictTrainTable <- table(svmPredictTrain$predictions,train_set[,c("Str_h_texture")])
svmPredictTest <- predict(svmClassifier,test_set[,-1],proba=TRUE,decisionValues=TRUE)
svmPredictTestTable <- table(svmPredictTest$predictions,test_set[,c("Str_h_texture")])

svmTestcol <- colnames(svmPredictTestTable)
svmTestrow <- rownames(svmPredictTestTable)

svmTraincol <- colnames(svmPredictTrainTable)
svmTrainrow <- rownames(svmPredictTrainTable)

sumElementinTable <- function(a,c,r){
  sum = 0
  for (i in c){
    if (i %in% r){
      sum = sum + a[i,i]
    }
  }
  return(sum)
}

svmPredictTestScore <- sumElementinTable(svmPredictTestTable,svmTestcol,svmTestrow)/sum(svmPredictTestTable)
svmPredictTrainScore <- sumElementinTable(svmPredictTrainTable,svmTraincol,svmTrainrow)/sum(svmPredictTrainTable)



#cannot run the below algorithms since it cannot allocate vector of size 16.5GB
#random forest is bad for sparse data which can be found in https://stats.stackexchange.com/questions/28828/is-there-a-random-forest-implementation-that-works-well-with-very-sparse-data
# RfClassifier = randomForest(Str_h_texture ~ .,data = train_set,proximity = T,mtry = 10)
# 
# rfTable <- table(predict(RfClassifier),train_set$Str_h_texture)
# 
# print(RfClassifier)
# plot(RfClassifier)

#In randomeForest() have tuneRF() for searching best optimal mtry values given for your data.
#We will depend on OOBError to define the most accurate mtry for our model which have the least OOBEError.
# x <- subset(train_set,select = -Str_h_texture)
# y <- train_set$Str_h_texture
# bestMtry <- tuneRF(x,y, stepFactor = 1.15, improve = 1e-5, ntree = 1000)



# the time of cart 0.320148 seconds
# the score of cart model 0.02476596
# Classification with CART model
cartFit <- rpart(Str_h_texture ~ .,data = train_set,control = rpart.control(cp = 0.0001))
#get cp value
printcp(cartFit)
#we can prune data with the CP value that contains the lowest error.

fit.pruned = prune(cartFit, cp = 0.00020393)

cartPrediction <- predict(fit.pruned, test_set, type = "vector")

data.frame(test_set,cartPrediction)


cartPrediction = round(cartPrediction,0)
cartTable <- table(test_set$Str_h_texture,cartPrediction)
cartrow <- rownames(cartTable)
cartcol <- colnames(cartTable)
cartscore <- sumElementinTable(cartTable,cartrow,cartcol)/sum(cartTable)
#classification with KNN model

#using knn now having some problem with knn
# knnClassifer <- knn(train_set,test_set,cl = train_set$Str_h_texture,k=9)
# 
# Kn_test <- table(test_set$Str_h_texture,knnClassifer)
# 
# #Full Data set can be used for cross validation
# knn.cross <- tune.knn(x = train_set, y = train_set$Str_h_texture, k = 2:20,tunecontrol=tune.control(sampling = "cross"), cross=10)
# #Summarize the resampling results set
# summary(knn.cross)
# 
# #give a warning which these variables have zero variances
# knnmodel <- train(
#   Str_h_texture ~., data = train_set, method = "knn",
#   trControl = trainControl("cv", number = 10),
#   preProcess = c("center","scale"),
#   tuneLength = 20
# )
# 
# plot()

#NaiveBayes classification
#NaiveBayes takes 4.305502 seconds
# the score is 0.02
# The formula is traditional Y~X1+X2+бн+Xn
# The data is typically a dataframe of numeric or factor variables.
# laplace provides a smoothing effect (as discussed below)
# subset lets you use only a selection subset of your data based on some boolean filter
# na.action lets you determine what to do when you hit a missing value in your dataset.

nbClassifier <- naiveBayes(as.factor(Str_h_texture) ~ .,data = train_set,laplace=2)
nbTestPrediction <- predict(nbClassifier,test_set,type = "class")
nbTableTest <- table(nbTestPrediction,test_set$Str_h_texture)

nbTestTablerow <- rownames(nbTableTest)
nbTestTablecol <- colnames(nbTableTest)
nbTestTablescore<- sumElementinTable(nbTableTest,nbTestTablerow,nbTestTablecol)/sum(nbTableTest)

nbTrainPrediction <- predict(nbClassifier,train_set,type = "class")
nbTrainTable <- table(nbTrainPrediction,train_set$Str_h_texture)

nbTrainTablerow <- rownames(nbTrainTable)
nbTrainTablecol <- colnames(nbTrainTable)
nbTrainTablescore <- sumElementinTable(nbTrainTable,nbTrainTablerow,nbTrainTablecol)




#fastnaivebayes takes 27.94227 seconds
# error is 0.9383002
dist <- fnb.detect_distribution(train_set.num_X)
gauss <- fnb.gaussian(train_set.num_X[,dist$gaussian], as.factor(train_set$Str_h_texture),sparse = TRUE,check = FALSE)
pred <- predict(gauss, train_set.num_X[,dist$gaussian])
error <- mean(as.factor(train_set$Str_h_texture)!=pred)




#neuro network
#We can us neuralnet() to train a NN model. Also, the train() function from caret can help us tune parameters.
#We can plot the result to see which set of parameters is fit our data the best.

Model <- train(Str_h_texture ~ .,
               data=train_set,
               method="neuralnet",
               ### Parameters for layers
               tuneGrid = expand.grid(.layer1=c(1:2), .layer2=c(0:2), .layer3=c(0)),
               ### Parameters for optmization
               learningrate = 0.01,
               threshold = 0.01,
               stepmax = 5000
)

# in nnclassifier y value should be normalized
train_set.norm <- train_set
maxStr_h_texture <- max(train_set.norm$Str_h_texture)
minStr_h_texture <- min(train_set.norm$Str_h_texture)
train_set.norm$Str_h_texture <- normalize(train_set.norm$Str_h_texture)
train_set.norm.X <- train_set.norm[,-1]

test_set.norm <- test_set
maxteStr_h_texture <- max(test_set.norm$Str_h_texture)
minteStr_h_texture <- min(test_set.norm$Str_h_texture)
test_set.norm$Str_h_texture <- normalize(test_set.norm$Str_h_texture)
test_set.norm.X <- test_set.norm[,-1]

nnClassifier <- neuralnet(Str_h_texture ~ .,data=train_set.norm, likelihood = TRUE, 
                          hidden = 1,linear.output = F,act.fct = "tanh")
print(nnClassifier$result.matrix)
plot(nnClassifier)

#prediction
output<- compute(nnClassifier,train_set[,-1])
p1 <- output$net.result
p1 <- p1 * (maxStr_h_texture-minStr_h_texture)
p1 <- round(p1,0)
nntable<-  table(train_set$Str_h_texture,p1)

#mlp  (similar to neural network) takes 58.56548 seconds
#The score of MLP is  0.02635098 
model <- mlp(train_set.norm.X, train_set.norm$Str_h_texture, size=5, learnFuncParams=c(0.1), 
             maxit=50, inputsTest=test_set.norm.X, targetsTest=test_set.norm$Str_h_texture)

summary(model)

predictions <- predict(model,test_set.norm.X)
predictions <- predictions * (maxteStr_h_texture - minteStr_h_texture)
predictions <- round(predictions,0)
mlptable <- table(test_set$Str_h_texture,predictions)
mlprow <-rownames(mlptable)
mlpcol <- colnames(mlptable)
mlpscore <- sumElementinTable(mlptable,mlprow,mlpcol)/sum(mlptable)



#Classification with the Adabag Boosting in R
#XGBOOSt cannot run within a specific time
adaClassifer <- boosting(as.factor(Str_h_texture) ~ .,data = train_set,boos = T,mfinal = 10)
adapred  <- predict(adaClassifer,test_set)
adaConfusionM <- adapred$confusion
adaError <- adapred$error

#Classification with xgbboost
xgb.train = xgb.DMatrix(data = as.matrix(train_set),label =as.matrix(train_set$Str_h_texture))
xgb.test = xgb.DMatrix(data = as.matrix(test_set),label = as.matrix(test_set$Str_h_texture))
validsoilTexture$Str_h_texture <- as.factor(validsoilTexture$Str_h_texture)
num_class = length(levels(validsoilTexture$Str_h_texture))

params = list(
  booster="gbtree",
  eta=0.001,
  max_depth=5,
  gamma=3,
  subsample=0.75,
  colsample_bytree=1,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=num_class+1
)

# Train the XGBoost classifer
xgb.fit=xgb.train(
  params=params,
  data=xgb.train,
  nrounds=10000,
  nthreads=1,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb.train,val2=xgb.test),
  verbose=0
)

xgb.fit

#lightGBM 
# score is 0.2914 the overhead of testing was 0.046899 seconds.

ltrain = lgb.Dataset(data = as.matrix(train_set.num_X),label = train_set$Str_h_texture, free_raw_data = FALSE)
params <- list(objective="regression", metric="l2")
model <- lgb.cv(params, 
                ltrain , 
                10, 
                nfold=5, 
                min_data=1, 
                learning_rate=1, 
                early_stopping_rounds=10,
                Depth = 5,
                lambda_l1 = 12,
                lambda_l2 = 13
)

ltest = lgb.Dataset.create.valid(ltrain , as.matrix(test_set.num_X), label = test_set$Str_h_texture)
valids <- list(test = ltest)

grid_search <- expand.grid(Depth = 1:8,
                           L1 = 8:16,
                           L2 = 8:16)

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



#catboost takes 4.908873 seconds
#The algorithm scores 0.02403448
fit_params <- list(l2_leaf_reg = 0.001,
                   depth=6,
                   learning_rate = 0.1,
                   iterations = 100,
                   random_seed = 233)


pool = catboost.load_pool(as.matrix(train_set.num_X), label = as.integer(train_set[,1]))

model <- catboost.train(pool, params = fit_params)




catprediction <- catboost.predict(model, 
                                  pool, 
                                  prediction_type = 'RawFormulaVal')


catprediction <- round(catprediction,0)

catTable <- table(train_set$Str_h_texture,catprediction)

catTablerow <- rownames(catTable)
catTablecol <- colnames(catTable)
catscore <- sumElementinTable(catTable,catTablerow,catTablecol)/sum(catTable)



