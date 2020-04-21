
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

#normalize the labr_value name
#preprocessing the data
normalize <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

featureSoilTable <- read.csv(file = "featureTable.csv")
#drop some of useless columns
featureSoilTable <- select (featureSoilTable,-c(X))

#change the NULL to na
featureSoilTable['h_texture'][featureSoilTable['h_texture'] == "NULL"] <- NA

#add appendix to colname:
colnames(featureSoilTable) <- paste("Str",colnames(featureSoilTable),sep = "_")
print(featureSoilTable)

#get the null value row
naValuerow <- which(is.na(featureSoilTable$Str_h_texture))

validsoilTexture <- featureSoilTable[-naValuerow,]
 
invalidsoilTexture <- featureSoilTable[naValuerow,]

# convert String to factor...
validsoilTexture$Str_h_texture <- factor(validsoilTexture$Str_h_texture)
validsoilTexture$Str_h_texture <- as.factor(validsoilTexture$Str_h_texture)

#change null value to 0
validsoilTexture[is.na(validsoilTexture)] = 0

ncol <- ncol(validsoilTexture)

#set random seed
set.seed(122)

#give the valid sample
split = sample.split(validsoilTexture$Str_h_texture,SplitRatio = 0.7)

train_set = subset(validsoilTexture, split == TRUE)
test_set = subset(validsoilTexture, split == FALSE)
  
#set all value to number
train_set.num = as.data.frame(sapply(train_set,as.factor))
train_set.num = as.data.frame(sapply(train_set,as.numeric))

test_set.num = as.data.frame(sapply(test_set, as.factor))
test_set.num = as.data.frame(sapply(test_set, as.numeric))

# Find the best model with the best cost parameter via 10-fold cross-validations
tryTypes=c(0:7)
tryCosts=c(1000,1,0.001)
bestCost=NA
bestAcc=0
bestType=NA

for(ty in tryTypes){
  for(co in tryCosts){
    acc=LiblineaR(data=train_set.num[,-1],target=train_set.num[,c("Str_h_texture")],type=ty,cost=co,bias=1,cross=5,verbose=FALSE)
    cat("Results for C=",co," : ",acc," accuracy.\n",sep="")
    if(acc>bestAcc){
      bestCost=co
      bestAcc=acc
      bestType=ty
    }
  }
}

# Sparse Logistic Regression
t=6

#svm classifier
svmClassifier <- LiblineaR(data = train_set.num[,-1],type=t,target = train_set.num[,c("Str_h_texture")])
svmPredictTest <- predict(svmClassifier,test_set.num[,-1])
svmPredictTestTable <- table(svmPredict$predictions,test_set.num[,c("Str_h_texture")])

#randomForest Classifier,error rate = 72.6%,random forest is bad for sparse data which can be found in https://stats.stackexchange.com/questions/28828/is-there-a-random-forest-implementation-that-works-well-with-very-sparse-data
RfClassifier = randomForest(Str_h_texture ~ .,data = train_set,ntree = 500,proximity = T,mtry = 10)

rfTable <- table(predict(RfClassifier),train_set$Str_h_texture)

print(RfClassifier)
plot(RfClassifier)

#In randomeForest() have tuneRF() for searching best optimal mtry values given for your data.
#We will depend on OOBError to define the most accurate mtry for our model which have the least OOBEError.
# x <- subset(train_set,select = -Str_h_texture)
# y <- train_set$Str_h_texture
# bestMtry <- tuneRF(x,y, stepFactor = 1.15, improve = 1e-5, ntree = 1000)

#Classification with CART model
cartFit <- rpart(Str_h_texture ~ .,data = train_set,control = rpart.control(cp = 0.0001))
#get cp value
printcp(cartFit)
#we can prune data with the CP value that contains the lowest error.

fit.pruned = prune(cartFit, cp = 0.007537688)

cartPrediction <- predict(fit.pruned, test_set, type = "class")

data.frame(test_set,cartPrediction)

confusionMatrix(test_set$Str_h_texture,cartPrediction)

#classification with KNN model

#preprocessing the data
normalize <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}
#normalize the training value
train_set.new<- as.data.frame(lapply(train_set[,-c(1)],normalize))

#replace null value with 0
train_set.new[is.na(train_set.new)] <- 0

test_set.new <- as.data.frame(lapply(test_set[,-c(1)],normalize))

test_set.new[is.na(test_set.new)] <- 0

ts <- train_set
ts$Str_h_texture <- as.numeric(ts$Str_h_texture)
train_set.norm <- as.data.frame(lapply(ts[,],normalize))
train_set.norm[is.na(train_set.norm)] <- 0

te <- test_set
te$Str_h_texture <- as.numeric(te$Str_h_texture)
test_set.norm <- as.data.frame(lapply(te[,],normalize))
test_set.norm[is.na(test_set.norm)] <- 0

#using knn
knnClassifer <- knn(train_set.new ,test_set.new,cl = train_set$Str_h_texture,k=9)

Kn_test <- table(test_set$Str_h_texture,knnClassifer)

Kn_TestScore = (sum(diag(Kn_test)))/sum(Kn_test)

cm_KnTestScore <- confusionMatrix(test_set$Str_h_texture,knnClassifer)

#Full Data set can be used for cross validation
knn.cross <- tune.knn(x = train_set.new, y = train_set$Str_h_texture, k = 2:20,tunecontrol=tune.control(sampling = "cross"), cross=10)
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

# Model <- train(Str_h_texture ~ .,     
#                data=train_set.norm,           
#                method="neuralnet",   
#                ### Parameters for layers
#                tuneGrid = expand.grid(.layer1=c(1:4), .layer2=c(0:4), .layer3=c(0)),               
#                ### Parameters for optmization
#                learningrate = 0.01,  
#                threshold = 0.01,     
#                stepmax = 50000         
# )

nnClassifier <- neuralnet(Str_h_texture ~ .,data=train_set.norm, likelihood = TRUE, 
                          hidden = c(1,2),linear.output = F)
print(nnClassifier$result.matrix)
plot(nnClassifier)

#prediction
output<- compute(nnClassifier,train_set.norm[,-1])
p1 <- output$net.result

#return back to string value

#get what predict value belongs to which label.
mulPred <- function(x){
  ma =  max(ts$Str_h_texture)
  mi =  min(ts$Str_h_texture)
  dif = ma - mi
  return (round(x * dif,digits = 0))
}

mp1 <- mulPred(p1)

l_ts = ts$Str_h_texture[!duplicated(ts$Str_h_texture)]
l_train_set = train_set$Str_h_texture[!duplicated(train_set$Str_h_texture)]

l_ts <- cbind(l_ts,l_train_set)
cvtFactorNum <- function(x){
  if (x>=1 & x<= nrow(l_ts)){
    return (l_train_set[x])
  }else{
    if (x<1){
      return (l_train_set[1])
    }else{
      return (l_train_set[nrow(l_ts)])
    }
  }
  
}

nnpred <- as.factor(unlist(lapply(mp1,cvtFactorNum)))
#confusion Matrix & Misclassifcation Error - training data
nnConfusionMatrix <-  confusionMatrix(train_set$Str_h_texture,nnpred)

#Classification with the Adabag Boosting in R
adaClassifer <- boosting(Str_h_texture ~ .,data = train_set,boos = T,mfinal = 10)
adapred  <- predict(adaClassifer,test_set)
adaConfusionM <- adapred$confusion
adaError <- adapred$error

#Classification with xgbboost
xgb.train = xgb.DMatrix(data = as.matrix(ts),label =as.matrix(ts$Str_h_texture)-1)
xgb.test = xgb.DMatrix(data = as.matrix(te),label = as.matrix(te$Str_h_texture)-1)

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
  num_class=num_class
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

