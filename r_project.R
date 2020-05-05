install.packages("openxlsx", dependencies = TRUE)
install.packages("tidyr")
install.packages("tidyverse")
install.packages("cluster")
install.packages("factoextra")
install.packages("factoextra")
install.packages("xgboost")
install.packages("mltools")
install.packages("DMwR")
install.packages("neuralnet")
install.packages("GGally")
install.packages("caret")
install.packages("DiagrammeR")
install.packages("openxlsx")
install.packages("party")
install.packages('nnet')

library(party)
library(DiagrammeR)
library(GGally)
library(neuralnet)
library(foreign)
library(plyr)
library(MASS)
library(rms)
library(ROCR)
library(pROC)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(e1071)
library(dplyr)
library(ggplot2)
library(tidyr)
library(tidyverse)
library(cluster)
library(factoextra)
library(xgboost)
library(Matrix)
library(mltools)
library(data.table)
library(DMwR)
library(zoo)
library(caret)
library(randomForest)
library(nnet)

####DATA HANDLING/INSPECTION
data = read.csv("heart.csv")
data = as.data.frame(data)
head(data,5)

data$sex = as.factor(data$sex) #gender
data$cp = as.factor(data$cp) #chest pain
data$fbs = as.factor(data$fbs) #fasting blood sugar
data$restecg = as.factor(data$restecg) #resting ECG
data$exang = as.factor(data$exang) # induced angina
data$slope = as.factor(data$slope) #slope
data$ca = as.factor(data$ca) #number of colored vessels
data$thal = as.factor(data$thal) #thal


str(data) #variables
summary(data) #variable statistics
sapply(data, sd)
apply(data, 2, function(x) {sum(is.na(x))}) #NAs inspection, 0 present
data_cv = data #cross-validation

n = nrow(data) #shuffling the data
data = data[sample(n),] 

rand_rows = sample(1:nrow(data), 0.6*nrow(data)) #split into training and validation dataset
train_data = data[rand_rows, ]
val_data = data[-rand_rows, ]

vars = colnames(data)[-length(colnames(data))] #creating a formula
y = "target" 
fmla = paste(y, paste(vars, collapse="+"),sep="~")
fmla

###LOGISTIC REGRESSION###
logit = glm(fmla, data = train_data, family = "binomial")
summary(logit)

prob_log = predict(logit, val_data) #prediction
pred_log = as.numeric(prob_log > 0.5)
val_data$pred_log = pred_log
confusionMatrix(table(val_data$target,val_data$pred_log), positive = "1")

auc_log = auc(val_data$target, val_data$pred_log) #auc
auc_log

plot(roc(val_data$target, val_data$pred_log, direction="<"),
     col="red", lwd=3, main="ROC")

err_log = mean(pred_log != val_data$target) #error
err_log

###DECISION TREES###
cart = rpart(fmla, data = train_data, method = 'class')
summary(cart)
rpart.plot(cart)

prob_cart = predict(cart, val_data) #prediction
pred_cart = as.numeric(prob_cart[,2] > 0.5)
val_data$pred_cart = pred_cart
confusionMatrix(table(val_data$target,val_data$pred_cart), positive = "1")

auc_cart = auc(val_data$target, val_data$pred_cart) #auc
auc_cart

plot(roc(val_data$target, val_data$pred_cart, direction="<"),
     col="red", lwd=3, main="ROC")

err_cart = mean(pred_cart != val_data$target) #error
err_cart

###NEURAL NETWORKS###
nn = nnet(target~ï..age+sex+cp+trestbps+chol+fbs+restecg+thalach
          +exang+oldpeak+slope+ca+thal, data = train_data, size = 5, decay = 5e-4, maxit = 100)
summary(nn)

prob_nn = predict(nn, val_data)
pred_nn = as.numeric(prob_nn > 0.5)
val_data$pred_nn = pred_nn
confusionMatrix(table(val_data$target,val_data$pred_nn), positive = "1")

auc_nn = auc(val_data$target, val_data$pred_nn) #auc
auc_nn

plot(roc(val_data$target, val_data$pred_nn, direction="<"),
     col="red", lwd=3, main="ROC")

err_nn = mean(pred_nn != val_data$target) #error
err_nn

###GRADIENT BOOSTING###
sparse_matrix = sparse.model.matrix(target ~ ., data = train_data)[,-1] #dummy contrast coding of categorical variables to fit the xgboost
sparse_matrix_val = sparse.model.matrix(target ~ ., data = val_data[,colnames(train_data)])[,-1]

labels = train_data$target

bst = xgboost(data = sparse_matrix, label = labels, max_depth = 4,
               eta = 1, nthread = 2, nrounds = 10,objective = "binary:logistic") #we should play with parameters

importance = xgb.importance(feature_names = colnames(sparse_matrix), model = bst) #importance of features
head(importance)

prob_xgboost = predict(bst, sparse_matrix_val)
pred_xgboost = as.numeric(prob_xgboost > 0.5)
val_data$pred_xgboost = pred_xgboost
confusionMatrix(table(val_data$target,val_data$pred_xgboost), positive = "1")

auc_xgboost = auc(val_data$target, val_data$pred_xgboost) #auc
auc_xgboost

plot(roc(val_data$target, val_data$pred_xgboost, direction="<"),
     col="red", lwd=3, main="ROC")

err_xgboost = mean(pred_xgboost != val_data$target) #error
err_xgboost

###RANDOM FOREST###
rf = randomForest(as.factor(target)~ï..age+sex+cp+trestbps+chol+fbs+restecg+thalach
                  +exang+oldpeak+slope+ca+thal, data = train_data,ntree = 33, nodesize =7, importance = T, proximity = TRUE)

pred_rf = predict(rf, val_data)
val_data$pred_rf = pred_rf
confusionMatrix(table(val_data$target,val_data$pred_rf), positive = "1")

auc_rf = auc(val_data$target,val_data$pred_rf) #auc
auc_rf

plot(roc(val_data$target, val_data$pred_rf, direction="<"),
     col="red", lwd=3, main="ROC")

err_rf = mean(pred_rf != val_data$target) #error
err_rf

###Cross-Validation
k = 100
acc_log = NULL
acc_cart = NULL
acc_nn = NULL
acc_xgboost = NULL
acc_rf = confusionMatrix(table(val_data$target,val_data$pred_rf), positive = "1")$overall['Accuracy']
er_log = NULL
er_cart = NULL
er_nn = NULL
er_xgboost = NULL
er_rf = mean(pred_rf != val_data$target)
ac_log = NULL
ac_cart = NULL
ac_nn = NULL
ac_xgboost = NULL
ac_rf = auc(val_data$target,as.numeric(val_data$pred_rf))

set.seed(100)

for (i in 1:k){ #sample randomly 100 times
  n = nrow(data_cv)
  data_temp = data_cv[sample(n),] 
  rand_rows = sample(1:nrow(data_temp), 0.7*nrow(data_temp)) #split into training and validation dataset
  train_data = data_temp[rand_rows, ]
  val_data = data_temp[-rand_rows, ]
  
  ##LOGISTISTIC
  logit = glm(fmla, data = train_data, family = "binomial")
  try({
    prob_log = predict(logit, val_data) #prediction
    pred_log = as.numeric(prob_log > 0.5)
    acc_log[i] = confusionMatrix(table(val_data$target,pred_log), positive = "1")$overall['Accuracy']
    er_log[i] = mean(pred_log != val_data$target) #error
    auc_log[i] = auc(val_data$target,val_data$pred_log)
  }, TRUE)
  ##DECISION TREE
  cart = rpart(fmla, data = train_data, method = 'class')
  try({
    prob_cart = predict(cart, val_data) #prediction
    pred_cart = as.numeric(prob_cart[,2] > 0.5)
    acc_cart[i] = confusionMatrix(table(val_data$target,pred_cart), positive = "1")$overall['Accuracy']
    er_cart[i] = mean(pred_cart != val_data$target) #error
    auc_cart[i] = auc(val_data$target,val_data$pred_cart)
  },TRUE)
  ##NEURAL NET
  nn = nnet(target~ï..age+sex+cp+trestbps+chol+fbs+restecg+thalach
            +exang+oldpeak+slope+ca+thal, data = train_data, size = 5, decay = 5e-4, maxit = 100)
  try({
    prob_nn = predict(nn, val_data)
    pred_nn = as.numeric(prob_nn > 0.5)
    acc_nn[i] = confusionMatrix(table(val_data$target,pred_nn), positive = "1")$overall['Accuracy']
    er_nn[i] = mean(pred_nn != val_data$target) #error
    auc_nn[i] = auc(val_data$target,val_data$pred_nn)
  },TRUE)
  ##BOOSTED TREE
  sparse_matrix = sparse.model.matrix(target ~ ., data = train_data)[,-1] #dummy contrast coding of categorical variables to fit the xgboost
  sparse_matrix_val = sparse.model.matrix(target ~ ., data = val_data[,colnames(train_data)])[,-1]
    
  labels = train_data$target
    
  bst = xgboost(data = sparse_matrix, label = labels, max_depth = 4,
                  eta = 1, nthread = 2, nrounds = 10,objective = "binary:logistic")
  try({
    prob_xgboost = predict(bst, sparse_matrix_val)
    pred_xgboost = as.numeric(prob_xgboost > 0.5)
    acc_xgboost[i] = confusionMatrix(table(val_data$target,pred_xgboost), positive = "1")$overall['Accuracy']
    er_xgboost[i] = mean(pred_xgboost != val_data$target) #error
    auc_xgboost[i] = auc(val_data$target,val_data$pred_xgboost)
  }, TRUE)
}

print(mean(acc_log, na.rm = TRUE))
print(mean(acc_cart, na.rm = TRUE))
print(mean(acc_nn, na.rm = TRUE))
print(mean(acc_xgboost, na.rm = TRUE))
print(acc_rf)

print(mean(er_log, na.rm = TRUE))
print(mean(er_cart, na.rm = TRUE))
print(mean(er_nn, na.rm = TRUE))
print(mean(er_xgboost, na.rm = TRUE))
print(er_rf)

print(mean(auc_log, na.rm = TRUE))
print(mean(auc_cart, na.rm = TRUE))
print(mean(auc_nn, na.rm = TRUE))
print(mean(auc_xgboost, na.rm = TRUE))
print(auc_rf)
