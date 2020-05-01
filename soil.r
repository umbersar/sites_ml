
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
normalize <- function(x){
  return (as.numeric((x-min(x))/(max(x)-min(x))))
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

#change null value to 0
validsoilTexture[is.na(validsoilTexture)] = 0

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

# test score around 63%

# Find the best model with the best cost parameter via 10-fold cross-validations
tryTypes=c(0:7)
tryCosts=c(1000,1,0.001)
bestCost=NA
bestAcc=0.6290723
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

#svm classifier
svmClassifier <- LiblineaR(data = train_set[,-1],target = train_set[,c("Str_h_texture")],bias=1,cost = 1000)
svmPredictTest <- predict(svmClassifier,test_set[,-1],proba=TRUE,decisionValues=TRUE)
svmPredictTestTable <- table(svmPredictTest$predictions,test_set[,c("Str_h_texture")])

svmcol <- colnames(svmPredictTestTable)
svmrow <- rownames(svmPredictTestTable)

sum = 0
for (i in svmcol){
  if (i %in% svmrow){
    sum = sum + svmPredictTestTable[i,i]
  }
}

svmPredictScore <- sum/sum(svmPredictTestTable)
#randomForest Classifier,error rate = 72.6%,random forest is bad for sparse data which can be found in https://stats.stackexchange.com/questions/28828/is-there-a-random-forest-implementation-that-works-well-with-very-sparse-data
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

#Classification with CART model geneating poor performance
cartFit <- rpart(Str_h_texture ~ .,data = train_set,control = rpart.control(cp = 0.0001))
#get cp value
printcp(cartFit)
#we can prune data with the CP value that contains the lowest error.

fit.pruned = prune(cartFit, cp = 0.007537688)

cartPrediction <- predict(fit.pruned, test_set, type = "class")

data.frame(test_set,cartPrediction)

confusionMatrix(test_set$Str_h_texture,cartPrediction)

#classification with KNN model

#using knn
knnClassifer <- knn(train_set,test_set,cl = train_set$Str_h_texture,k=9)

Kn_test <- table(test_set$Str_h_texture,knnClassifer)

Kn_TestScore = (sum(diag(Kn_test)))/sum(Kn_test)

cm_KnTestScore <- confusionMatrix(test_set$Str_h_texture,knnClassifer)

#Full Data set can be used for cross validation
knn.cross <- tune.knn(x = train_set, y = train_set$Str_h_texture, k = 2:20,tunecontrol=tune.control(sampling = "cross"), cross=10)
#Summarize the resampling results set
summary(knn.cross)


#NaiveBayes classification
# The formula is traditional Y~X1+X2+бн+Xn
# The data is typically a dataframe of numeric or factor variables.
# laplace provides a smoothing effect (as discussed below)
# subset lets you use only a selection subset of your data based on some boolean filter
# na.action lets you determine what to do when you hit a missing value in your dataset.

nbClassifier <- naiveBayes(Str_h_texture ~ .,data = train_set,laplace=2)
nbTestPrediction <- predict(nbClassifier,test_set,type = "class")
nbTableTest <- table(nbTestPrediction,test_set$Str_h_texture)
confusionMatrix(nbTableTest)

nbTrainPrediction <- predict(nbClassifier,train_set,type = "class")
nbTrainTable <- table(nbTrainPrediction,train_set$Str_h_texture)
confusionMatrix(nbTrainTable )

#neuro network
set.seed(222)

#We can us neuralnet() to train a NN model. Also, the train() function from caret can help us tune parameters.
#We can plot the result to see which set of parameters is fit our data the best.

Model <- train(Str_h_texture ~ .,
               data=train_set,
               method="neuralnet",
               ### Parameters for layers
               tuneGrid = expand.grid(.layer1=c(1:4), .layer2=c(0:4), .layer3=c(0)),
               ### Parameters for optmization
               learningrate = 0.01,
               threshold = 0.01,
               stepmax = 50000
)

nnClassifier <- neuralnet(Str_h_texture ~ .,data=train_set, likelihood = TRUE, 
                          hidden = c(1,2),linear.output = F)
print(nnClassifier$result.matrix)
plot(nnClassifier)

#prediction
output<- compute(nnClassifier,train_set[,-1])
p1 <- output$net.result

nnConfusionMatrix <-  confusionMatrix(train_set$Str_h_texture,nnpred)

#Classification with the Adabag Boosting in R
adaClassifer <- boosting(Str_h_texture ~ .,data = train_set,boos = T,mfinal = 10)
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
train_set.num_X <- select (train_set,-c(Str_h_texture))
test_set.num_X <- select (test_set,-c(Str_h_texture))

ltrain = lgb.Dataset(data = as.matrix(train_set.num_X),label = train_set$Str_h_texture, free_raw_data = FALSE)
params <- list(objective="regression", metric="l2")
model <- lgb.cv(params, 
                ltrain , 
                10, 
                nfold=5, 
                min_data=1, 
                learning_rate=1, 
                early_stopping_rounds=10,
                Depth = 8,
                lambda_l1 = 10,
                lambda_l2 = 10
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

