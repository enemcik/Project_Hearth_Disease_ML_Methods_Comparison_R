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

#### alternative way using ROCR package####
library(ROCR)
prob_log = predict(logit, val_data, type = "response")
pred = prediction(prob_log, val_data$target) #create a prediction object with true values and predicted ones

roc.perf = performance(pred, measure = "tpr", x.measure = "fpr") #performance measure as ROC
plot(roc.perf)

auc.perf = performance(pred, measure = "auc") #performance measure AUC
auc.perf@y.values

opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}
print(opt.cut(roc.perf, pred)) #formula which finds "optimal" cutoff weighting both sensitivity and specificity equally (TPR and FPR)

acc.perf = performance(pred, measure = "acc") # find best cutoff according to accuracy
plot(acc.perf)

ind = which.max( slot(acc.perf, "y.values")[[1]] )
acc = slot(acc.perf, "y.values")[[1]][ind]
cutoff = slot(acc.perf, "x.values")[[1]][ind]
print(c(accuracy= acc, cutoff = cutoff))

#### END alternative way using ROCR package####

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
nn = nnet(target~?..age+sex+cp+trestbps+chol+fbs+restecg+thalach
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
rf = randomForest(as.factor(target)~?..age+sex+cp+trestbps+chol+fbs+restecg+thalach
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
  pPos<-sum(train_data$target)/nrow(train_data) #Get unconditional probablity of target being 1
  
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
  nn = nnet(target~?..age+sex+cp+trestbps+chol+fbs+restecg+thalach
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

### Alternative for-loop with ROCR package solution for logit #### 

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
logit_pred = vector(mode = "list", length = 100)
prob_log = vector(mode = "list", length = 100)

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
    prob_log[[i]] = predict(logit, val_data, type = "response")
    logit_pred[[i]] = val_data$target # save prediction and true value for later use in a list
  }, TRUE)
}

ind.drop= unlist(lapply(prob_log, is.numeric)) # create index for dropping NULL values as they interfer with later analysis
prob_log= prob_log[ind.drop==TRUE] # drop NULL values (created because new factor levels in test set)
logit_pred = logit_pred[ind.drop==TRUE] # drop true values corresponding to NULL values

manypred = prediction(prob_log, logit_pred)

many.roc.perf = performance(manypred, measure = "tpr", x.measure = "fpr")
plot(many.roc.perf, col=1:10)
abline(a=0, b= 1) #plot of all the ROC curves

print(opt.cut(many.roc.perf, manypred)) # find optimal cutoffs for each estimation

many.acc.perf = performance(manypred, measure = "acc") # find cutoff for highest possible accuracy
sapply(manypred@labels, function(x) mean(x == 1))

mapply(function(x, y){
  ind = which.max( y )
  acc = y[ind]
  cutoff = x[ind]
  return(c(accuracy= acc, cutoff = cutoff))
}, slot(many.acc.perf, "x.values"), slot(many.acc.perf, "y.values"))

manypauc.perf = performance(manypred, measure = "auc") # area under the curve values for all estimations
manypauc.perf@y.values
