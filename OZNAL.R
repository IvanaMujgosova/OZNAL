

#precision pre kazdu cinnost do grafu
#vysledky prvotnych experimentov, a zobrazit ich
#popisat SVM klasifikator

#Load dataset
library(datasets)

dataset <- read.table( "Xtrain.csv", sep=";", header=FALSE)

dataset_2 <- read.table( "X_test.csv", sep=";", header=FALSE)



#pomenovania stlpcov
columns.names <- read.table( "features.txt", sep=" ", header=FALSE)
column.names2 <- as.vector(columns.names$V2)
colnames(dataset) <- column.names2
colnames(dataset_2) <- column.names2

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
unique_dataset_2 <- subset(dataset_2, select=which(!duplicated(names(dataset_2))))


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

#---------------------------------------------------------------------------------

# ensure the results are repeatable
set.seed(56)

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

#RANDOM FOREST - to uz je hotove inde :)
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
HARTrainData = sample(1:6000,1:380)
HARValData = sample(6801:7300,1:477)

hist(reduced_dataset$ACTIVITY)


# support vector machine, dalsi pokus... - ZATIAL SLABOTA!
library(kernlab)
rbf <- rbfdot(sigma=0.1)
harSVM <- ksvm(ACTIVITY~.,data=reduced_dataset[HARTrainData,],type="C-bsvc",kernel=rbf,C=10,prob.model=TRUE)
fitted(harSVM)
gg <- predict(harSVM, reduced_dataset[HARValData,-84], type="probabilities")

plot(gg)

# support vector machine, dalsi pokus, s celym datasetom
library(kernlab)
rbf <- rbfdot(sigma=0.05)
harSVM_2 <- ksvm(ACTIVITY~.,data=unique_dataset[,-479],type="C-bsvc",kernel=rbf,C=10,prob.model=TRUE)
fitted(harSVM_2)
gg <- predict(harSVM_2, unique_dataset_2, type="probabilities")

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

#
#--------------------------------------------------------------------------------------------------------
#                                               SVM 2.0                                                 |
#                                                                                                       |
#--------------------------------------------------------------------------------------------------------
# Druhy balik pre SVM

install.packages("e1071")
library("e1071")

Testset <- reduced_dataset[1:6000, ]
Trainset <- reduced_dataset[6001:7000, ]

Model <- svm(ACTIVITY~., data = Testset)
Prediction <- predict (Model, Trainset[-84])
Prediction = floor(Prediction)

Tab <- table(pred=Prediction, true=Trainset[,84])

options(max.print=1000000) 


write.csv(Tab, file = "Tab.csv")

#PRINCIPAL COMPONENT ANALYSIS

# log transform 
log.data <- log(unique_dataset[,1:477])
data.activity <- unique_dataset[, 478]

# Standardizing each variable
unique_dataset.scaled <- data.frame(apply(unique_dataset[,1:477], 2, scale))

data.scaled <- data.frame(t(na.omit(t(unique_dataset.scaled))))

# apply PCA - scale. = TRUE is highly 
# advisable, but default is FALSE. 

#PRVY VYSLEDOK PCA
data.pca.2 <- prcomp(data.scaled, retx=TRUE)

#DRUHY VYSLEDOK PCA
data.pca <- prcomp(data.scaled,
                 center = TRUE,
                 scale. = TRUE) 

#vysledek
print(data.pca.2)
print(data.pca)

# grafy
plot(data.pca, type = "l")
plot(data.pca.2, type = "l")

# summary method
s1 <- summary(data.pca)
s2 <- summary(data.pca.2)

s1$importance

#*************************************************************

dim(unique_dataset)

Eigenvalues <- eigen(cov(unique_dataset[,1:477]))$values
Eigenvectors <- eigen(cov(unique_dataset[,1:477]))$vectors
PC <- as.matrix(unique_dataset[,1:477]) %*% Eigenvectors

print(round(Eigenvalues/sum(Eigenvalues) * 100, digits = 2))
round(cumsum(Eigenvalues)/sum(Eigenvalues) * 100, digits = 2)

data.pca.3 <- prcomp(unique_dataset[,1:477])
data.pca.3.var <- data.pca.3$sdev^2

data.pca.3.var[1:3]
Eigenvalues[1:3]

plot(data.pca.3)


#------------------------------------------------------------------------------------
#Parallel SVM
#------------------------------------------------------------------------------------
install.packages("parallelSVM")

train = read.csv("Letrain.csv")
test = read.csv("Letest.csv")

Names <- names(train)

Train_0 = grep("[M|m]ean",Names)
train1 = train[,c(Train_0,563)]
test1 = test[,c(Train_0,563)]

library(e1071)
library(parallelSVM)
svm.fit = parallelSVM(Activity~. ,
                      samplingSize = 0.6,
                      scale = TRUE,
                      type = "C-classification",
                      seed = 25,
                      cross = 15,
                      coef0 = 2,
                      cost = 40,
                      tolerance = 0.002,
                      epsilon = 0.2,
                      data = train)
summary(svm.fit)


prediction = predict(svm.fit,test0)

plot(test0$Activity)
plot(train0$Activity)


