library(caret) 
library(kernlab)
library(randomForest)
library(corrplot)
library(foreach)
library(doParallel)

if (!file.exists("pml-testing.csv")){
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
}

if (!file.exists("pml-training.csv")){
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
}

set.seed(23)

## Loading data
data= read.csv("pml-training.csv", sep=",",na.strings = c("NA", ""))

## Check the number of variables to reduce to non NAs columns.
dim(data)
## Remove variables with all NA

variables = sapply(data, function(x){sum(is.na(x))})
data = data[,which(variables == 0)]

##  Only 60 out of 160 variables are considered. The rest columns have all NA's Values.
dim(data)

## Reamaining variables
str(data)

## Create training and test partitions
inTrain = createDataPartition(y=data$classe, p=0.75, list=FALSE)
training = data[inTrain,]
testing = data[-inTrain,]

training = training[,8:60]
training$classe <-as.factor(training$classe)
testing = testing[,8:60]
testing$classe <-as.factor(testing$classe)


training.scale<- scale(training[,-53],center=TRUE,scale=TRUE);

testing.scale<- scale(testing[,-53],center=TRUE,scale=TRUE);

corMat <- cor(training.scale)

corrplot(corMat, order = "hclust")

highlyCor <- findCorrelation(corMat, 0.5)
#Apply correlation filter at 0.70,
#then we remove all the variable correlated with more 0.7.
datMyFiltered.scale <- training.scale[,-highlyCor]
datTestingMyFiltered.scale <- testing.scale[,-highlyCor]

corMatFiltered <- cor(datMyFiltered.scale)
corrplot(corMatFiltered, order = "hclust")

training = data.frame(datMyFiltered.scale, training[,53])

testing = data.frame(datTestingMyFiltered.scale, testing[,53])

colnames(training)[22] = "classe"
colnames(testing)[22] = "classe"


registerDoParallel()
model <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
  randomForest(x=training[,-22], y=training$classe, ntree=ntree)
}


pred = predict(model,training)
confusionMatrix(pred,training$classe)


predictedValues = predict(model, testing)

confusionMatrix(predictedValues, testing$classe)
