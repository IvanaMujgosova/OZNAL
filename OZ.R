
#http://www.heatonresearch.com/2013/06/12/r-classification.html

#precision pre kazdu cinnost do grafu
#vysledky prvotnych experimentov, a zobrazit ich
#popisat SVM klasifikator

#Load dataset
library(datasets)

dataset <- read.table( "xtrain.csv", sep=";", header=FALSE)


#pomenovania stlpcov
columns.names <- read.table( "features.txt", sep=" ", header=FALSE)
column.names2 <- as.vector(columns.names$V2)
colnames(dataset) <- column.names2

#pridanie stlpca s aktivitou
activity_column <- read.table( "ytrain.csv", sep=";", header=FALSE)
activity_vector <- activity_column["V1"]


#pridanie stlpca ktory to je subject
subject_column <- read.table( "subjecttrain.csv", sep=";", header=FALSE)
subject_vector <- subject_column["V1"]
class(subject_vector)
class(activity_vector)

subject_vector <- as.vector(t(subject_vector))
activity_vector <- as.vector(t(activity_vector))


#v datasete su duplikaty v stlpcoch, remove it :
unique_dataset <- subset(dataset, select=which(!duplicated(names(dataset))))

#pouzit do vizualizacie
summary(unique_dataset)
plot(unique_dataset[1:400,1:7])
plot(unique_dataset[1:400,8:14])
plot(unique_dataset[1:400,15:21])
plot(unique_dataset[1:400,115:121])
hist(unique_dataset$ACTIVITY)

unique_dataset <- grep("[M|m]ean",columns.names)

print(idxTrain0)

#add two columns to dataset
unique_dataset$ACTIVITY <- activity_vector
unique_dataset$SUBJECT <- subject_vector

write.csv(unique_dataset,'train_all.csv')


#---------------------------------------------------------------------------------

# ensure the results are repeatable
set.seed(7)

# load the library
library(mlbench)
library(caret)

# load the data
data(dataset)


# calculate correlation matrix
correlationMatrix <- cor(unique_dataset[,1:477])

# summarize the correlation matrix
print(correlationMatrix)

# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)

# print indexes of highly correlated attributes
print(highlyCorrelated)

write.csv(reduced_dataset,'reduced_dataset.csv')

#indexy vysoko korelovanych stlpcov
indices.to.delete <- as.vector(highlyCorrelated)

#redukovany dataset - bez vysoko korelovanych stlpcov
reduced_dataset <- unique_dataset[,-indices.to.delete] 

print(reduced_dataset)

#-------------------------------------------------------------------------------------------

#Toto funguje len na binarny dataset (my nemame binarny)

# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)

# train the model
model <- train(ACTIVITY~., data=unique_dataset, method="lvq", preProcess="scale", trControl=control)

# estimate variable importance
importance <- varImp(model, scale=FALSE)

# summarize importance
print(importance)

# plot importance
plot(importance)

#-----------------------------------------------------------------------------------------------

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)

# run the RFE algorithm
results <- rfe(reduced_dataset[1:20,1:121], reduced_dataset[1:20,123], sizes=c(1:121), rfeControl=control)

# summarize the results
print(results)

# list the chosen features
predictors(results)

# plot the results
plot(results, type=c("g", "o"))

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-++-+-+-+-+-+-+-+-+-+

control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# SVM
set.seed(7)
fit.svm <- train(ACTIVITY~., data=unique_dataset[1:200,1:479], method="svmRadial", trControl=control)
print(fit.svm)

colnames(reduced_dataset)[1] <- "tBodyAccMeanY"
colnames(reduced_dataset)[2] <- "tBodyAccMeanZ"


#building the classification tree
#install if necessary
install.packages("tree")
library(tree)
tree1 <- tree(ACTIVITY ~ tBodyAccMeanY + tBodyAccMeanZ , data = reduced_dataset)
summary(tree1)

tree(formula = ACTIVITY ~ tBodyAccMeanY + tBodyAccMeanZ , data = reduced_dataset)

plot(tree1)
text(tree1)

#rozdelila som si train dataset na train a val
HARTrainData = sample(1:3500,300)
HARValData = sample(3501:7300,100)

hist(reduced_dataset$ACTIVITY)


# support vector machine, dalsi pokus... - ZATIAL SLABOTA!
library(kernlab)
rbf <- rbfdot(sigma=0.1)
harSVM <- ksvm(ACTIVITY~.,data=reduced_dataset[HARTrainData,],type="C-bsvc",kernel=rbf,C=10,prob.model=TRUE)
fitted(harSVM)
gg <- predict(harSVM, reduced_dataset[HARValData,-84], type="probabilities")

plot(gg)

#Neural Networks
library(nnet)
ideal <- class.ind(reduced_dataset$ACTIVITY)
#trening nn
harANN = nnet(reduced_dataset[HARTrainData,-84], ideal[HARTrainData,], size=10, softmax=TRUE)
predict(harANN, reduced_dataset[HARValData,-84], type="class")

cisla <- scan('shit2.csv', what=numeric(), sep=",", quiet=TRUE)
av = mean(cisla)
av


ciselka <- read.table( "porbabilities.csv", sep=";", header=FALSE)
graf <- sapply(ciselka[2:101,], mean, na.rm=TRUE)
plot(graf)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#random forest - pokus c. 1 - 97,5%

install.packages("randomForest")

library(randomForest)

rfNews()

datasetReduced <- read.table( "reduced_dataset2.csv", sep=",", header=TRUE)

#data(dataset)

#print (reduced_dataset)

print(head(datasetReduced))

#randomForest(formula, dataset)

output.forest <- randomForest(ACTIVITY ~ tBodyAcc_meanY + tBodyAcc_meanZ + tBodyAcc_arCoeffX4 + tBodyAcc_arCoeffY4 + tBodyAcc_arCoeffZ4 + tBodyAcc_correlationXY + tBodyAcc_correlationXZ + tGravityAcc_sma + tGravityAcc_energyY + tGravityAcc_energyZ + tGravityAcc_iqrX + tGravityAcc_entropyY + tGravityAcc_entropyZ + tGravityAcc_arCoeffX1 + tGravityAcc_arCoeffY4 + tGravityAcc_correlationXY + tGravityAcc_correlationXZ + tGravityAcc_correlationYZ + tBodyAccJerk_meanX + tBodyAccJerk_meanY + tBodyAccJerk_meanZ + tBodyAccJerk_arCoeffX2 + tBodyAccJerk_arCoeffX4 + tBodyAccJerk_arCoeffY4 + tBodyAccJerk_arCoeffZ2 + tBodyAccJerk_arCoeffZ4 + tBodyAccJerk_correlationXY + tBodyAccJerk_correlationXZ + tBodyAccJerk_correlationYZ + tBodyGyro_meanY + tBodyGyro_meanZ + tBodyGyro_arCoeffZ4 + tBodyGyro_correlationXY + tBodyGyro_correlationXZ + tBodyGyro_correlationYZ + tBodyGyroJerk_meanY + tBodyGyroJerkm_meanZ + tBodyGyroJerk_arCoeffX3 + tBodyGyroJerk_arCoeffX4 + tBodyGyroJerk_arCoeffY3 + tBodyGyroJerk_arCoeffY4 + tBodyGyroJerk_arCoeffZ4 + tBodyGyroJerk_correlationXY + BodyGyroJerk_correlationXZ + tBodyGyroJerk_correlationYZ + tGravityAccMag_arCoeff4 + tBodyAccJerkMag_arCoeff3 + tBodyGyroMag_arCoeff4 + tBodyGyroJerkMag_arCoeff3 + fBodyAcc_minY + fBodyAcc_minZ + fBodyAcc_maxIndsX + fBodyAcc_maxIndsY + fBodyAcc_maxIndsZ + fBodyAcc_skewnessZ + fBodyAcc_bandsEnergy57_64 + fBodyAccJerk_minY + fBodyAccJerk_minZ + fBodyAccJerk_maxIndsY + fBodyAccJerk_maxIndsZ + fBodyAccJerk_kurtosisX + fBodyAccJerk_kurtosisY + fBodyAccJerk_kurtosisZ + fBodyAccJerk_bandsEnergy57_64 + fBodyGyr_minY + fBodyGyr_minZ + fBodyGyro_maxIndsX + fBodyGyro_maxIndsY + fBodyGyro_maxIndsZ + fBodyGyro_meanFreqX + fBodyGyro_kurtosisX + fBodyGyro_skewnessZ + fBodyGyro_bandsEnergy57_64 + fBodyAccMag_maxInds + fBodyAccMag_skewness + fBodyBodyAccJerkMag_maxInds + fBodyBodyGyroMag_maxInds + fBodyBodyGyroMag_kurtosis + fBodyBodyGyroJerkMag_maxInds + fBodyBodyGyroJerkMag_meanFreq + fBodyBodyGyroJerkMag_kurtosis + angle_tBodyAccMean_gravity + angle_tBodyGyroJerkMean_gravityMean,
                              data = datasetReduced)

output.forest <- randomForest(ACTIVITY ~ tBodyAcc_meanY + tBodyAcc_meanZ + tBodyAcc_arCoeffX4 + tBodyAcc_arCoeffY4 + tBodyAcc_arCoeffZ4 + tBodyAcc_correlationXY,
                              data = datasetReduced)

print(output.forest)

print(importance(fit,type = 2)) 

plot(output.forest)


install.packages("party")
library("party")
x <- ctree(Species ~ ., data=output.forest)
plot(x, type="simple")

varImpPlot(output.forest)

plot(output.forest, output.forest)
abline(c(0,1),col=2)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#test data
library(datasets)

dataset <- read.table( "test_data.csv", sep=";", header=FALSE)


#pomenovania stlpcov
columns.names <- read.table( "features_test.txt", sep=" ", header=FALSE)
column.names2 <- as.vector(columns.names$V2)
colnames(dataset) <- column.names2

#pridanie stlpca s aktivitou
activity_column <- read.table( "ytest.csv", sep=";", header=FALSE)
activity_vector <- activity_column["V1"]


#pridanie stlpca ktory to je subject
subject_column <- read.table( "subjecttest.csv", sep=";", header=FALSE)
subject_vector <- subject_column["V1"]
class(subject_vector)
class(activity_vector)

subject_vector <- as.vector(t(subject_vector))
activity_vector <- as.vector(t(activity_vector))


#v datasete su duplikaty v stlpcoch, remove it :
unique_dataset <- subset(dataset, select=which(!duplicated(names(dataset))))

#pouzit do vizualizacie
summary(unique_dataset)
plot(unique_dataset[1:400,1:7])
plot(unique_dataset[1:400,8:14])
plot(unique_dataset[1:400,15:21])
plot(unique_dataset[1:400,115:121])
hist(unique_dataset$ACTIVITY)



#add two columns to dataset
unique_dataset$ACTIVITY <- activity_vector
unique_dataset$SUBJECT <- subject_vector

write.csv(dataset,'dataset_test2.csv')

#---------------------------------------------------------------------------------



# ensure the results are repeatable
set.seed(7)

# load the library
library(mlbench)
library(caret)

# load the data
data(dataset)


# calculate correlation matrix
correlationMatrix <- cor(unique_dataset[,1:477])

# summarize the correlation matrix
print(correlationMatrix)

# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)

# print indexes of highly correlated attributes
print(highlyCorrelated)



#indexy vysoko korelovanych stlpcov
indices.to.delete <- as.vector(highlyCorrelated)

#redukovany dataset - bez vysoko korelovanych stlpcov
reduced_dataset <- unique_dataset[,-indices.to.delete] 

print(reduced_dataset)

write.csv(reduced_dataset,'reduced_dataset_test.csv')



#--------------------------------------------------------------------

#test data with activity
library(datasets)

dataset <- read.table( "test_data.csv", sep=";", header=FALSE)


#pomenovania stlpcov
columns.names <- read.table( "features_test.txt", sep=" ", header=FALSE)
column.names2 <- as.vector(columns.names$V2)
colnames(dataset) <- column.names2

#pridanie stlpca s aktivitou
activity_column <- read.table( "ytest.csv", sep=";", header=FALSE)
activity_vector <- activity_column["V1"]


#pridanie stlpca ktory to je subject
subject_column <- read.table( "subjecttest.csv", sep=";", header=FALSE)
subject_vector <- subject_column["V1"]
class(subject_vector)
class(activity_vector)

subject_vector <- as.vector(t(subject_vector))
activity_vector <- as.vector(t(activity_vector))


#v datasete su duplikaty v stlpcoch, remove it :
unique_dataset <- subset(dataset, select=which(!duplicated(names(dataset))))

#pouzit do vizualizacie
summary(unique_dataset)
plot(unique_dataset[1:400,1:7])
plot(unique_dataset[1:400,8:14])
plot(unique_dataset[1:400,15:21])
plot(unique_dataset[1:400,115:121])
hist(unique_dataset$ACTIVITY)



#add two columns to dataset
unique_dataset$ACTIVITY <- activity_vector
unique_dataset$SUBJECT <- subject_vector

write.csv(unique_dataset,'test_dataset_labels_activity.csv')


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#random forest - pokus c. 2 - 97,5% test

install.packages("randomForest")

library(randomForest)

rfNews()

datasetReduced <- read.table( "test_data_with_labels.csv", sep=",", header=TRUE)

#data(dataset)

#print (reduced_dataset)

print(head(datasetReduced))

#randomForest(formula, dataset)

output.forest <- randomForest(ACTIVITY ~ tBodyAcc_meanY + tBodyAcc_meanZ + tBodyAcc_arCoeffX4 + tBodyAcc_arCoeffY4 + tBodyAcc_arCoeffZ4 + tBodyAcc_correlationXY + tBodyAcc_correlationXZ + tGravityAcc_sma + tGravityAcc_energyY + tGravityAcc_energyZ + tGravityAcc_iqrX + tGravityAcc_entropyY + tGravityAcc_entropyZ + tGravityAcc_arCoeffX1 + tGravityAcc_arCoeffY4 + tGravityAcc_correlationXY + tGravityAcc_correlationXZ + tGravityAcc_correlationYZ + tBodyAccJerk_meanX + tBodyAccJerk_meanY + tBodyAccJerk_meanZ + tBodyAccJerk_arCoeffX2 + tBodyAccJerk_arCoeffX4 + tBodyAccJerk_arCoeffY4 + tBodyAccJerk_arCoeffZ2 + tBodyAccJerk_arCoeffZ4 + tBodyAccJerk_correlationXY + tBodyAccJerk_correlationXZ + tBodyAccJerk_correlationYZ + tBodyGyro_meanY + tBodyGyro_meanZ + tBodyGyro_arCoeffZ4 + tBodyGyro_correlationXY + tBodyGyro_correlationXZ + tBodyGyro_correlationYZ + tBodyGyroJerk_meanY + tBodyGyroJerkm_meanZ + tBodyGyroJerk_arCoeffX3 + tBodyGyroJerk_arCoeffX4 + tBodyGyroJerk_arCoeffY3 + tBodyGyroJerk_arCoeffY4 + tBodyGyroJerk_arCoeffZ4 + tBodyGyroJerk_correlationXY + BodyGyroJerk_correlationXZ + tBodyGyroJerk_correlationYZ + tGravityAccMag_arCoeff4 + tBodyAccJerkMag_arCoeff3 + tBodyGyroMag_arCoeff4 + tBodyGyroJerkMag_arCoeff3 + fBodyAcc_minY + fBodyAcc_minZ + fBodyAcc_maxIndsX + fBodyAcc_maxIndsY + fBodyAcc_maxIndsZ + fBodyAcc_skewnessZ + fBodyAcc_bandsEnergy57_64 + fBodyAccJerk_minY + fBodyAccJerk_minZ + fBodyAccJerk_maxIndsY + fBodyAccJerk_maxIndsZ + fBodyAccJerk_kurtosisX + fBodyAccJerk_kurtosisY + fBodyAccJerk_kurtosisZ + fBodyAccJerk_bandsEnergy57_64 + fBodyGyr_minY + fBodyGyr_minZ + fBodyGyro_maxIndsX + fBodyGyro_maxIndsY + fBodyGyro_maxIndsZ + fBodyGyro_meanFreqX + fBodyGyro_kurtosisX + fBodyGyro_skewnessZ + fBodyGyro_bandsEnergy57_64 + fBodyAccMag_maxInds + fBodyAccMag_skewness + fBodyBodyAccJerkMag_maxInds + fBodyBodyGyroMag_maxInds + fBodyBodyGyroMag_kurtosis + fBodyBodyGyroJerkMag_maxInds + fBodyBodyGyroJerkMag_meanFreq + fBodyBodyGyroJerkMag_kurtosis + angle_tBodyAccMean_gravity + angle_tBodyGyroJerkMean_gravityMean,
                              data = datasetReduced)



output.forest <- randomForest(ACTIVITY ~ tBodyAcc_meanY + tBodyAcc_meanZ + tBodyAcc_arCoeffX4 + tBodyAcc_arCoeffY4 + tBodyAcc_arCoeffZ4 + tBodyAcc_correlationXY,
                              data = datasetReduced)

print(output.forest)

print(importance(fit,type = 2)) 

plot(output.forest)


install.packages("party")
library("party")
x <- ctree(Species ~ ., data=output.forest)
plot(x, type="simple")

varImpPlot(output.forest)

plot(output.forest, output.forest)
abline(c(0,1),col=2)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#random forest pokus c. 2
trainData <- read.table( "reduced_dataset2.csv", sep=",", header=TRUE)
testData <- read.table( "test_dataset_labels_activity.csv", sep=",", header=TRUE)

library(randomForest)

activity_rf <- randomForest(ACTIVITY~.,data=trainData,ntree=100,proximity=TRUE)
table(predict(activity_rf),trainData$ACTIVITY)

print(activity_rf)

plot(activity_rf)

importance(activity_rf)

varImpPlot(activity_rf)

irisPred<-predict(activity_rf,newdata=testData)
table(irisPred, testData$ACTIVITY)

irisPred$aggregate
irisPred$individual


plot(margin(activity_rf,testData$ACTIVITY))

tune.rf <- tuneRF(columns.names,columns.names, stepFactor=0.5)

print(irisPred)

plot(irisPred)

write.csv(table(irisPred, testData$ACTIVITY),'RF_pred.csv')

write.csv(table(irisPred$individual, testData$ACTIVITY),'RF_pred2.csv')


+-----------------------------------------------------
# RF 3
  
trainData <- read.table( "reduced_dataset2.csv", sep=",", header=TRUE)
testData <- read.table( "test_dataset_labels_activity.csv", sep=",", header=TRUE)

library(randomForest)

head (trainData)

barplot(table(trainData$ACTIVITY))

trainData$ACTIVITY[trainData$ACTIVITY == 1] <- 'WALKING 1'
trainData$ACTIVITY[trainData$ACTIVITY == 2] <- 'WALKING_UPSTAIRS 2'
trainData$ACTIVITY[trainData$ACTIVITY == 3] <- 'WALKING_DOWNSTAIRS 3'
trainData$ACTIVITY[trainData$ACTIVITY == 4] <- 'SITTING 4'
trainData$ACTIVITY[trainData$ACTIVITY == 5] <- 'STANDING 5'
trainData$ACTIVITY[trainData$ACTIVITY == 6] <- 'LAYING 6'
trainData$ACTIVITY <- as.factor(trainData$ACTIVITY)

table(trainData$ACTIVITY)

model <- randomForest(ACTIVITY ~ . -angle_tBodyGyroJerkMean_gravityMean, data = trainData)

model

pred <- predict(model, newdata = testData)
table(pred, testData$ACTIVITY)

barplot(table(testData$ACTIVITY))

table(testData$ACTIVITY)

plot(testData$ACTIVITY, col="black",pch="o")

points(pred, col="red", pch="*")


legend(1,100,legend=c("original","RF"), col=c("black","red"),
       pch=c("o","*"),lty=c(1,2), ncol=1)


#------------------------------------------------
#all data
trainData <- read.table( "train_all.csv", sep=",", header=TRUE)

model <- randomForest(ACTIVITY ~ . -SUBJECT, data = unique_dataset)

#----------------------------
train = read.csv("train_all.csv")
Names <- names(train)
idxTrain0 = grep("[M|m]ean",Names)
train2 = train[,c(idxTrain0,480)]

#pomenovania stlpcov
columns.names <- read.table( "features.txt", sep=" ", header=FALSE)
column.names2 <- as.vector(columns.names$V2)
colnames(dataset) <- column.names2

#pridanie stlpca s aktivitou
activity_column <- read.table( "ytrain.csv", sep=";", header=FALSE)
activity_vector <- activity_column["V1"]


#pridanie stlpca ktory to je subject
subject_column <- read.table( "subjecttrain.csv", sep=";", header=FALSE)
subject_vector <- subject_column["V1"]
class(subject_vector)
class(activity_vector)

subject_vector <- as.vector(t(subject_vector))
activity_vector <- as.vector(t(activity_vector))


#v datasete su duplikaty v stlpcoch, remove it :
unique_dataset <- subset(dataset, select=which(!duplicated(names(dataset))))

#pouzit do vizualizacie
summary(unique_dataset)
plot(unique_dataset[1:400,1:7])
plot(unique_dataset[1:400,8:14])
plot(unique_dataset[1:400,15:21])
plot(unique_dataset[1:400,115:121])
hist(unique_dataset$ACTIVITY)

unique_dataset <- grep("[M|m]ean",columns.names)

print(idxTrain0)

#add two columns to dataset
train2$ACTIVITY <- activity_vector
unique_dataset$SUBJECT <- subject_vector

write.csv(unique_dataset,'train_all.csv')


# test

#Load dataset
library(datasets)

dataset <- read.table( "xtest.csv", sep=";", header=FALSE)


#pomenovania stlpcov
columns.names <- read.table( "features.txt", sep=" ", header=FALSE)
column.names2 <- as.vector(columns.names$V2)
colnames(dataset) <- column.names2

#pridanie stlpca s aktivitou
activity_column <- read.table( "ytest.csv", sep=";", header=FALSE)
activity_vector <- activity_column["V1"]


#pridanie stlpca ktory to je subject
subject_column <- read.table( "subjecttest.csv", sep=";", header=FALSE)
subject_vector <- subject_column["V1"]
class(subject_vector)
class(activity_vector)

subject_vector <- as.vector(t(subject_vector))
activity_vector <- as.vector(t(activity_vector))


#v datasete su duplikaty v stlpcoch, remove it :
unique_dataset <- subset(dataset, select=which(!duplicated(names(dataset))))

#pouzit do vizualizacie
summary(unique_dataset)
plot(unique_dataset[1:400,1:7])
plot(unique_dataset[1:400,8:14])
plot(unique_dataset[1:400,15:21])
plot(unique_dataset[1:400,115:121])
hist(unique_dataset$ACTIVITY)

#add two columns to dataset
unique_dataset$ACTIVITY <- activity_vector
unique_dataset$SUBJECT <- subject_vector

write.csv(unique_dataset,'test_all2.csv')

test = read.csv("test_all2.csv")
Names <- names(test)
idxTest0 = grep("[M|m]ean",Names)
test2 = test[,c(idxTrain0,480)]


#•++++++++++++++++++++++++++++++++++++++++
#RF 4

trainData <- read.table( "reduced_dataset2.csv", sep=",", header=TRUE)
testData <- read.table( "test_dataset_labels_activity.csv", sep=",", header=TRUE)

library(randomForest)

head (trainData)

barplot(table(trainData$ACTIVITY))

trainData$ACTIVITY[trainData$ACTIVITY == 1] <- 'WALKING 1'
trainData$ACTIVITY[trainData$ACTIVITY == 2] <- 'WALKING_UPSTAIRS 2'
trainData$ACTIVITY[trainData$ACTIVITY == 3] <- 'WALKING_DOWNSTAIRS 3'
trainData$ACTIVITY[trainData$ACTIVITY == 4] <- 'SITTING 4'
trainData$ACTIVITY[trainData$ACTIVITY == 5] <- 'STANDING 5'
trainData$ACTIVITY[trainData$ACTIVITY == 6] <- 'LAYING 6'
trainData$ACTIVITY <- as.factor(trainData$ACTIVITY)

table(trainData$ACTIVITY)

model <- randomForest(ACTIVITY ~ . , data = train2)

model

pred <- predict(model, newdata = test2)
table(pred, testData$ACTIVITY)

barplot(table(test2$ACTIVITY))

table(testData$ACTIVITY)

plot(testData$ACTIVITY, col="black",pch="o")

points(pred, col="red", pch="*")


legend(1,100,legend=c("original","RF"), col=c("black","red"),
       pch=c("o","*"),lty=c(1,2), ncol=1)

