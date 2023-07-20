#****************************************#
#            Charlotte Abrams            #
#             Data Science II            #
#               Homework 1               #
#               Spring 2019              #
#****************************************#

#install.packages("glmnet")
#install.packages("pls")

library(caret)
library(boot)
library(glmnet)
library(pls)

#Import Test Data
testData <- read.csv("~/Documents/Charlotte/Columbia/Spring 2019/DSII/HW1/Data/solubility_test.csv")

#Import Training Data
trainData <- read.csv("~/Documents/Charlotte/Columbia/Spring 2019/DSII/HW1/Data/solubility_train.csv")

#Omit rows containing missing data
testData <- na.omit(testData)
trainData <- na.omit(trainData)

# Train data matrix
xTrain <- model.matrix(Solubility~.,trainData)[,-1]

#Train data response vector
yTrain <- trainData$Solubility

#Test data matrix
xTest <- model.matrix(Solubility~.,testData)[,-1]

#Test data response vector
yTest <- testData$Solubility

#  A  ###############################################################################################
set.seed(123)

#Fit linear model on the training data
model = lm(Solubility~., data = trainData)

#Fit trainning model on test data set
pred = predict(model, testData)

#Calculate accuracy with MSE
MSE = mean((yTest - pred)^2)
print(MSE)
#[1] 0.5558898

#  B  ###############################################################################################

# Use cross-validation
set.seed(123)
cv.ridge <- cv.glmnet(xTrain, yTrain,
                       alpha = 0)
plot(cv.ridge)

best.lambda <- cv.ridge$lambda.min
print(best.lambda)
#[1] 0.14784

#Create training model using ridge regression (alpha=0) with best lambda
ridge.model <- glmnet(xTrain,yTrain, 
                      alpha = 0, 
                      lambda = best.lambda)

mat.coef <- coef(ridge.model)
dim(mat.coef)
#[1] 229 1

#Fit trainning model on test data set
pred <- predict(ridge.model, s = best.lambda, newx = xTest)
print(pred)

#Calculate accuracy with MSE
MSE <- mean((pred - yTest)^2)
print(MSE)
#[1] 0.5147447

#  C  ###############################################################################################

set.seed(123)

#Create training model using lasso regression (alpha=1) with best lambda
cv.lasso <- cv.glmnet(xTrain, yTrain,
                       alpha = 1)

best.lambda <- cv.lasso$lambda.min
print(best.lambda)
# [1] 0.004621057
plot(cv.lasso)

#Create training model using lasso regression (alpha=1) with best lambda
lasso.model <- glmnet(xTrain, yTrain,
                alpha = 1,
                lambda = best.lambda)

#Fit trainning model on test data set
pred <- predict(lasso.model, s = best.lambda, newx = xTest)
print(pred)

#Calculate accuracy with MSE
MSE <- mean((pred - yTest)^2)
print(MSE)
# [1] 0.4952673

#Lasso Non-Zero Coefficients
#Retrieving the lasso coefficients
lasso.coef <- predict(lasso.model,type = "coefficients", s = best.lambda)[1:length(lasso.model$beta),]

#Printing non-zero coefficients
lasso.coef[lasso.coef != 0]
#There are 142 non-zero coefficients

#  D  ###############################################################################################
set.seed(123)

pcr.model <- pcr(Solubility ~ ., 
                 data = trainData, 
                 scale = TRUE, 
                 validation = "CV")

summary(pcr.model)

predy2.pcr <- predict(pcr.model, testData, 
                      ncomp = 228)

#Calculate accuracy with MSE
MSE = mean((predy2.pcr - yTest)^2)
print(MSE)
#[1] 0.5558898 and M=228

#  E  ###############################################################################################
#The Mean Square Error (MSE) is the smallest when ussing the Lasso Regression model (MSE=0.4952673), 
#therefore, it is the best model to use for this analysis. The models with the highest MSE values and 
#the worst models for this adanlysis would be the PCR and linear models both with MSE=0.5558

