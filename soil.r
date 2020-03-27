

#add data limit
memory.limit(15000000000)

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

#reduce the amount to 10000
soil <- head(soil,n=1000L)
labmCode = soil$labm_code
  
#factorize the soil
labmNum <- factor(c(labmCode))

soil$labm_code <- as.numeric(labmNum)

#change labr_value to numeric
soil$labr_value <- as.numeric(soil$labr_value)


#load the "h_soil_water_stat" non-null column
#fill in the NULL value as NA value
soil$h_soil_water_stat[soil$h_soil_water_stat == "NULL"] = NA

#get the null value row
naValuerow <- which(is.na(soil$h_soil_water_stat))

validsoilSample <- soil[-naValuerow,]

invalidsoilSample <- soil[naValuerow,]

soil_label = validsoilSample$h_soil_water_stat
soil_label <- factor(c(soil_label))
validsoilSample$h_soil_water_stat <- as.numeric(soil_label)

#set random seed
set.seed(122)

#give the valid sample
validsoilSample <- validsoilSample[,c("h_soil_water_stat","labm_code","labr_value")]


split = sample.split(validsoilSample$h_soil_water_stat,SplitRatio = 0.75)
train_set = subset(validsoilSample, split == TRUE)
test_set = subset(validsoilSample, split ==FALSE)

# x <- subset(validsoilSample, select=-h_soil_water_stat)
# y <- validsoilSample$h_soil_water_stat

#to check if which value is the best
# svm_tune <- tune(svm, train.x=x, train.y=y, 
#                  kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
# 
# print(svm_tune)


#svm classifier
svmClassifier = svm(formula = h_soil_water_stat ~ .,
                 data = train_set,
                 kernel = 'radial',
                 cost = 0.1,
                 gamma = 2)

train_set$h_soil_water_stat <- as.character(train_set$h_soil_water_stat)
train_set$h_soil_water_stat <- as.factor(train_set$h_soil_water_stat)
test_set$h_soil_water_stat <- as.character(test_set$h_soil_water_stat)
test_set$h_soil_water_stat <- as.factor(test_set$h_soil_water_stat)
# Predicting the Test set results 
y_Svm_test_pred <- predict(svmClassifier,newdata = test_set[,c("labm_code","labr_value")])
y_Svm_train_pred <- predict(svmClassifier,newdata = train_set[,c("labm_code","labr_value")])

cm_SVM1 = table(test_set[,c("h_soil_water_stat")],y_Svm_test_pred)
cm_SVM2 = table(train_set[,c("h_soil_water_stat")],y_Svm_train_pred)

#randomForest Classifier
RfClassifier = randomForest(h_soil_water_stat ~ .,data = train_set,ntree = 10,proximity = T)

rfTable <- table(predict(RfClassifier),train_set$h_soil_water_stat)
print(RfClassifier)
plot(RfClassifier)

#Classification with CART model
cartFit <- rpart(h_soil_water_stat ~ .,method = "class",data = train_set,minsplit=3,minbucket = 1)
fancyRpartPlot(cartFit, caption = NULL)
print(cartFit)
cartPrediction <- predict(cartFit,test_set)
# confusionMatrix(test_set$h_soil_water_stat, cartPrediction)

