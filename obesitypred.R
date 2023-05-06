#Remove all objects from the current workspace
rm(list=ls())
#Install all the needed packages, i.e. install.packages("pROC") etc if such package is 
not installed yet. Afterwards, load all the needed packages.
library(caret)
library(MASS)
library(tidyverse)
library(pROC)
library(corrplot)
library(earth)
library(rpart)
library(gplots)
library(ipred)
##############################################################################
# R Codes for Supervised Machine Learning Prediction of Non-Communicable Diseases#
##############################################################################
#STEP 0:Read in the data into the R enviroment:
odata <- read_csv("C:/Users/OEA/Desktop/ADAGlobalConcept/Projects/odata.csv")
#View(odata)
#head(odata)
dim(odata)
#STEP 1:Split the data into training and test set:
#Split the data into training (70%) and test set (30%)
set.seed(25034)
training.samples <- odata$overweight %>%
  createDataPartition(p = 0.7, list = FALSE)
train.data <- odata[training.samples, ]
test.data <- odata[-training.samples, ]
#Re-label values of outcomes (1 = overweight, 0 = underweight)
train.data$overweight[train.data$overweight ==0] <- "underweight"
train.data$overweight[train.data$overweight ==1] <- "overweight"
test.data$overweight[test.data$overweight ==0] <- "underweight"
test.data$overweight[test.data$overweight ==1] <- "overweight"
#convert outcome variable to type factor
train.data$overweight <- as.factor(train.data$overweight)
test.data$overweight <- as.factor(test.data$overweight)
###############################
#Set the training control scheme
ctrl <- trainControl(method ="repeatedcv", number = 10, repeats = 5, 
savePredictions = "all", sampling = "down")
#Support Vector Machine - svm
#Fit the model
set.seed(250301)
model.svm <- train(overweight ~ .,
                   data = train.data, method = "svmLinear",
                   preProcess = c("center"), tunelenght = 5,
                   trControl = ctrl)
#Predict the outcome using model1 from train.data applied to thew test.data
predictions.svm <- predict(model.svm, newdata = test.data)
#create ConfusionMatrix
confusionMatrix(data = predictions.svm, test.data$overweight, mode = "everything")
######### Logistic Regression ##########
set.seed(250303)
model.lr <- train(overweight ~ .,
                  data = train.data, method = "glm",
                  family = "binomial", preProcess = c("scale"), 
                  trControl = ctrl)

#Predict the outcome using model1 from train.data applied to thew test.data
predictions.lr <- predict(model.lr, newdata = test.data)

#create ConfusionMatrix
confusionMatrix(data = predictions.lr, test.data$overweight, mode = "everything")  
######### MARS ##########
set.seed(250304)
model.mars <- train(overweight ~ .,
                    data = train.data, method = "earth",
                    preProcess = c("scale"), 
                    trControl = ctrl)

#Predict the outcome using model1 from train.data applied to thew test.data
predictions.mars <- predict(model.mars, newdata = test.data)

#create ConfusionMatrix
confusionMatrix(data = predictions.mars, test.data$overweight, mode = "everything")
######### Naive Bayes ##########
set.seed(2503025)
model.nb <- train(overweight ~ .,
                  data = train.data, method = "nb",
                  preProcess = c("scale"), 
                  trControl = ctrl)
#Predict the outcome using model1 from train.data applied to thew test.data
predictions.nb <- predict(model.nb, newdata = test.data)
#create ConfusionMatrix
confusionMatrix(data = predictions.nb, test.data$overweight, mode = "everything")
######### Tree ##########
set.seed(250306)
model.tree <- train(overweight ~ .,
                    data = train.data, method = "rpart",
                    preProcess = c("scale"), 
                    trControl = ctrl)

#Predict the outcome using model1 from train.data applied to thew test.data
predictions.tree <- predict(model.tree, newdata = test.data)
#create ConfusionMatrix
confusionMatrix(data = predictions.tree, test.data$overweight, mode = "everything")
######### Linear Discriminant Analysis ##########
set.seed(250307)
model.lda <- train(overweight ~ .,
                   data = train.data, method = "lda",
                   preProcess = c("scale"), 
                   trControl = ctrl)

#Predict the outcome using model1 from train.data applied to thew test.data
predictions.lda <- predict(model.lda, newdata = test.data)
#create ConfusionMatrix
confusionMatrix(data = predictions.lda, test.data$overweight, mode = "everything")
######### KNN ##########
set.seed(250308)
model.knn <- train(overweight ~ .,
                   data = train.data, method = "knn",
                   preProcess = c("scale"), 
                   trControl = ctrl)
plot(model.knn)
#Predict the outcome using model1 from train.data applied to thew test.data
predictions.knn <- predict(model.knn, newdata = test.data)
table(predictions.knn)
#create ConfusionMatrix
confusionMatrix(data = predictions.knn, test.data$overweight, mode = "everything")
######### Random Forest ##########
set.seed(250309)
model.rf <- train(overweight ~ .,
                  data = train.data, method = "rf",
                  preProcess = c("scale"), 
                  trControl = ctrl)

#Predict the outcome using model1 from train.data applied to thew test.data
predictions.rf <- predict(model.rf, newdata = test.data)
#create ConfusionMatrix
confusionMatrix(data = predictions.rf, test.data$overweight, mode = "everything")
######### Bagging ##########
set.seed(250310)
initial.bag <- bagging(overweight ~ ., data = train.data, nbagg = 100, 
coob = TRUE, control = rpart.control(minsplit = 2, cp = 0))
model.bag <- train(overweight ~ .,
                   data = train.data, method = "treebag", nbagg = 200,
                   preProcess = c("scale"), 
                   control = rpart.control(minsplit = 2, cp = 0),
                   trControl = ctrl)
#Predict the outcome using model1 from train.data applied to thew test.data
predictions.bag <- predict(model.bag, newdata = test.data)
#create ConfusionMatrix
confusionMatrix(data = predictions.bag, test.data$overweight, mode = "everything")
####### Boosting ###############
set.seed(250302)
training.samples <- odata$overweight %>%
  createDataPartition(p = 0.7, list = FALSE)
btrain.data <- odata[training.samples, ]
btest.data <- odata[-training.samples, ]
model.boost <- gbm(overweight ~ ., data = btrain.data, n.trees = 500, 
distribution = "bernoulli", interaction.depth = 1, cv.folds = 50, shrinkage = 0.2)
best_n_tress <- which.min(model.boost$cv.error)
# summary(model.boost)
boost.predict.train <- predict(model.boost, newdata = btrain.data, 
                               n.trees = best_n_tress, type = "response")
boost.train.error <- mean((boost.predict.train - btrain.data$overweight)^2)
boost.predict.test <- predict(model.boost, newdata = btest.data, 
                              n.trees = best_n_tress, type = "response")
boost.test.error <- mean((boost.predict.test - btest.data$overweight)^2)

boost.pred.test.label <- ifelse(boost.predict.test > 0.5, '1', '0')
table.boosting <- table(true = btest.data$overweight, predicted = boost.pred.test.label)
confusionMatrix(table.boosting, mode = "everything")
###########################################################
#######################Metric Analysis#####################
models = c("SVM", "LR", "MARS", "NB", "Tree", "LDA", "KNN", "RF", "Bagging","GB")
#############ACURRACY ANALYSIS ###############
acc = c(0.6468, 0.6567, 0.6119, 0.7463, 0.6269, 0.6816, 0.6716, 0.5920, 0.6119, 0.7861)
#par(mfrow = c(1,2))
plot(acc, xaxt = "n", xlab = "Algorithms", type = "b", 
ylab = "Accuracy", col = 3, lwd = 5)
axis(side = 1, at = seq(1,10), labels = models)
###############################################
############# BALANCED ACCURACY ANALYSIS ###############
bacc = c(0.6511, 0.6216, 0.5730, 0.5431, 0.6255, 0.6808, 0.6741, 0.5716, 0.5851, 0.7684)
plot(bacc, xaxt = "n", xlab = "Algorithms", type = "b", 
ylab = "Balanced Accuracy", col = 4, lwd = 5)
axis(side = 1, at = seq(1,10), labels = models)
#############SENSITIVITY ANALYSIS ###############
sen = c(0.6604, 0.5472, 0.4906, 0.1132, 0.6226, 0.6792, 0.6792, 0.5283, 0.5283, 0.7868)
plot(sen, xaxt = "n", xlab = "Algorithms", type = "b", 
ylab = "Sensitivity", col = 2, lwd = 5)
axis(side = 1, at = seq(1,10), labels = models)
###############################################
############# F1 Score ANALYSIS ###############
f1 = c(0.4965, 0.4567, 0.4000, 0.1905, 0.4681, 0.5294, 0.5217, 0.4058, 0.4179, 0.8782)
plot(f1, xaxt = "n", xlab = "Algorithms", type = "b", 
ylab = "F1 Score", col = 5, lwd = 5)
axis(side = 1, at = seq(1,10), labels = models)
###############################################
