
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

#another svm library
library(liquidSVM)

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

#a pca method for 
pca <- PCA(validsoilTexture[2:108])
summary(pca)
#set random seed
set.seed(122)

#give the valid sample
split = sample.split(validsoilTexture$Str_h_texture,SplitRatio = 0.7)
train_set = subset(validsoilTexture, split == TRUE)
test_set = subset(validsoilTexture, split == FALSE)

# tuned to find out the best cost and gamma
tuned <- tune.svm(Str_h_texture ~.,
                  data = train_set,
                  gamma = 10^(-3:3), cost = 10^(-3:3),scale = FALSE,
                  na.action = na.omit,
                  tunecontrol = tune.control(cross = 10))

summary(tuned)

#svm classifier
svmClassifier = svm(formula = Str_h_texture ~ .,
                    data = train_set,
                    type="C-classification",
                    kernel = 'polynomial',
                    cost = 1000,
                    gamma = 1000,
                    scale = FALSE)

print(svmClassifier)
summary(svmClassifier)


# Predicting the Test set results 
y_Svm_test_pred <- predict(svmClassifier,newdata = test_set[-1])
y_Svm_train_pred <- predict(svmClassifier,newdata = train_set[-1])

cm_SVMTest= table(test_set[,c("Str_h_texture")],y_Svm_test_pred)
cm_SVMTrain = table(train_set[,c("Str_h_texture")],y_Svm_train_pred)

#compute the score
#NOTICE: Although trainscore is good but testscore seems not very good.
cm_SVMTestScore <- (sum(diag(cm_SVMTest)))/sum(cm_SVMTest)
cm_SVMTrainScore <-  (sum(diag(cm_SVMTrain)))/sum(cm_SVMTrain)

#*Another SVM method through liquidSVM

#randomForest Classifier,error rate = 72.6%
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

