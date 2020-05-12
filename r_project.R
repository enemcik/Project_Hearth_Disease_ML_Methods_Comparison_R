install.packages(c("parallel","parallelMap"))
library(parallel)
library(parallelMap)
#library(party)
#library(DiagrammeR)
library(GGally)
library(neuralnet)
#library(foreign)
library(plyr)
library(MASS)
library(rms)
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
library(ROCR)
library(mlr)

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

set.seed(935) #for reproducibility

n = nrow(data) #shuffling the data
data = data[sample(n),] 

rand_rows = sample(1:nrow(data), 0.6*nrow(data)) #split into training and validation dataset
train_data = data[rand_rows, ]
val_data = data[-rand_rows, ]

vars = colnames(data)[-length(colnames(data))] #creating a formula
y = "target" 
fmla = paste(y, paste(vars, collapse="+"),sep="~")
fmla

######################
####Cross-validated###
######################

k = 100
acc_log = NULL
acc_cart = NULL
acc_nn = NULL
acc_xgboost = NULL
er_log = NULL
er_cart = NULL
er_nn = NULL
er_xgboost = NULL
auc_log = NULL
auc_cart = NULL
auc_nn = NULL
auc_xgboost = NULL
cut_log = NULL
cut_cart = NULL
cut_nn = NULL
cut_xgboost = NULL
cut_rf = NULL

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
    prob_log = predict(logit, val_data, type = "response")
    pred_log = ROCR::prediction(prob_log, val_data$target) #create a prediction object with true values and predicted ones
    
    roc.perf_log = performance(pred_log, measure = "tpr", x.measure = "fpr") #performance measure as ROC
    
    auc.perf_log = performance(pred_log, measure = "auc") #performance measure AUC
    auc_log[i] = auc.perf_log@y.values[[1]]
    
    acc.perf_log = performance(pred_log, measure = "acc") # find best cutoff according to accuracy
    
    ind_log = which.max( slot(acc.perf_log, "y.values")[[1]] )
    acc_log[i] = slot(acc.perf_log, "y.values")[[1]][ind_log]
    cutoff_log = slot(acc.perf_log, "x.values")[[1]][ind_log]
    cut_log[i] = cutoff_log
    
    pred_log = as.numeric(prob_log > cutoff_log) #ERROR- minimized by maximizing ACCURACY using its cutoff
    er_log[i] = mean(pred_log != val_data$target)
  },TRUE)
 
  ##DECISION TREE - BAGGED
  
  m<-dim(train_data)[1]
  ntree<-33
  samples<-sapply(1:ntree,
                  FUN = function(iter)
                  {sample(1:m, size=m, replace=T)}) #replace=T makes it a bootstrap 
  try({
    treelist<-lapply(1:ntree, #Training the individual decision trees and return them in a list
                     FUN=function(iter) {
                       samp <- samples[,iter];
                       rpart(fmla,train_data[samp,])
                     }
    )
    #predict.bag assumes the underlying classifier returns decision probabilities, not decisions. 
    predict.bag<-function(treelist,newdata) {
      preds<-sapply(1:length(treelist),
                    FUN=function(iter) {
                      predict(treelist[[iter]],newdata=newdata)
                    }
      )
      predsums<-rowSums(preds)
      predsums/length(treelist)
    }
    
    prob_bagtree = predict.bag(treelist, newdata=val_data)
    pred_bagtree = ROCR::prediction(prob_bagtree, val_data$target)
    
    roc.perf_bagtree = performance(pred_bagtree, measure = "tpr", x.measure = "fpr") #performance measure as ROC
    
    auc.perf_bagtree = performance(pred_bagtree, measure = "auc") #performance measure AUC
    auc_cart[i] = auc.perf_bagtree@y.values[[1]]
    
    acc.perf_bagtree = performance(pred_bagtree, measure = "acc") # find best cutoff according to accuracy
    
    ind_bagtree = which.max( slot(acc.perf_bagtree, "y.values")[[1]] )
    acc_cart[i] = slot(acc.perf_bagtree, "y.values")[[1]][ind_bagtree]
    cutoff_bagtree = slot(acc.perf_bagtree, "x.values")[[1]][ind_bagtree]
    cut_cart[i] = cutoff_bagtree
    
    pred_bagtree = as.numeric(prob_bagtree > cutoff_bagtree) #ERROR- minimized by maximizing ACCURACY using its cutoff
    er_cart[i] = mean(pred_bagtree != val_data$target)
  },TRUE)
  
  ##NEURAL NET
  nn = nnet(target~ï..age+sex+cp+trestbps+chol+fbs+restecg+thalach
            +exang+oldpeak+slope+ca+thal, data = train_data, size = 5, decay = 5e-4, maxit = 100)
  try({
    prob_nn = predict(nn, val_data)
    pred_nn = ROCR::prediction(prob_nn, val_data$target) #create a prediction object with true values and predicted ones
    
    roc.perf_nn = performance(pred_nn, measure = "tpr", x.measure = "fpr") #performance measure as ROC
    
    auc.perf_nn = performance(pred_nn, measure = "auc") #performance measure AUC
    auc_nn[i] = auc.perf_nn@y.values[[1]]
    
    acc.perf_nn = performance(pred_nn, measure = "acc") # find best cutoff according to accuracy
    
    ind_nn = which.max( slot(acc.perf_nn, "y.values")[[1]] )
    acc_nn[i] = slot(acc.perf_nn, "y.values")[[1]][ind_nn]
    cutoff_nn = slot(acc.perf_nn, "x.values")[[1]][ind_nn]
    cut_nn[i] = cutoff_nn
    
    pred_nn = as.numeric(prob_nn > cutoff_nn) #ERROR- minimized by maximizing ACCURACY using its cutoff
    er_nn[i] = mean(pred_nn != val_data$target)
  },TRUE)

  ##BOOSTED TREE
  sparse_matrix = sparse.model.matrix(target ~ ., data = train_data)[,-1] #dummy contrast coding of categorical variables to fit the xgboost
  sparse_matrix_val = sparse.model.matrix(target ~ ., data = val_data[,colnames(train_data)])[,-1]
  
  labels = train_data$target
  
  bst = xgboost(data = sparse_matrix, label = labels, max_depth = 4,
                eta = 1, nthread = 2, nrounds = 10,objective = "binary:logistic")

  importance = xgb.importance(feature_names = colnames(sparse_matrix), model = bst) #importance of features
  head(importance)
  try({
    prob_xgboost = predict(bst, sparse_matrix_val)
    
    pred_xgboost = ROCR::prediction(prob_xgboost, val_data$target) #create a prediction object with true values and predicted ones
    
    roc.perf_xgboost = performance(pred_xgboost, measure = "tpr", x.measure = "fpr") #performance measure as ROC
    
    auc.perf_xgboost= performance(pred_xgboost, measure = "auc") #performance measure AUC
    auc_xgboost[i] = auc.perf_xgboost@y.values[[1]]
    
    acc.perf_xgboost = performance(pred_xgboost, measure = "acc") # find best cutoff according to accuracy
    
    ind_xgboost = which.max( slot(acc.perf_xgboost, "y.values")[[1]] )
    acc_xgboost[i] = slot(acc.perf_xgboost, "y.values")[[1]][ind_xgboost]
    cutoff_xgboost = slot(acc.perf_xgboost, "x.values")[[1]][ind_xgboost]
    cut_xgboost[i] = cutoff_xgboost
    
    pred_xgboost = as.numeric(prob_xgboost > cutoff_xgboost) #ERROR- minimized by maximizing ACCURACY using its cutoff
    er_xgboost[i] = mean(pred_xgboost != val_data$target)
  },TRUE)
}

###RANDOM FOREST### have to talk about the ntree and etc settings
rf = randomForest(as.factor(target)~ï..age+sex+cp+trestbps+chol+fbs+restecg+thalach
                  +exang+oldpeak+slope+ca+thal, data = train_data,ntree = 33, nodesize =7, importance = T, proximity = TRUE)

prob_rf = predict(rf, val_data,type='prob') # returns probabilities not 0,1 values with type="prob"

pred_rf = ROCR::prediction(prob_rf[,2], val_data$target) #create a prediction object with true values and predicted ones

roc.perf_rf = performance(pred_rf, measure = "tpr", x.measure = "fpr") #performance measure as ROC
plot(roc.perf_rf)

auc.perf_rf= performance(pred_rf, measure = "auc") #performance measure AUC
auc_rf = auc.perf_rf@y.values[[1]]

print(opt.cut(roc.perf_rf, pred_rf)) #formula which finds "optimal" cutoff weighting both sensitivity and specificity equally (TPR and FPR)

acc.perf_rf = performance(pred_rf, measure = "acc") # find best cutoff according to accuracy
plot(acc.perf_rf)

ind_rf = which.max( slot(acc.perf_rf, "y.values")[[1]] )
acc_rf= slot(acc.perf_rf, "y.values")[[1]][ind_rf]
cutoff_rf = slot(acc.perf_rf, "x.values")[[1]][ind_rf]
cut_rf = cutoff_rf
print(c(accuracy= acc_rf, cutoff = cutoff_rf))

pred_rf = as.numeric(prob_rf > cutoff_rf) #ERROR- minimized by maximizing ACCURACY using its cutoff
er_rf = mean(pred_rf != val_data$target)


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

print(mean(cut_log, na.rm = TRUE))
print(mean(cut_cart, na.rm = TRUE))
print(mean(cut_nn, na.rm = TRUE))
print(mean(cut_xgboost, na.rm = TRUE))
print(cut_rf)

#######################
####Replicated Paper###
#######################

###LOGISTIC REGRESSION###
logit = glm(fmla, data = train_data, family = "binomial")
summary(logit)

prob_log = predict(logit, val_data, type = "response")
pred_log = prediction(prob_log, val_data$target) #create a prediction object with true values and predicted ones

roc.perf_log = performance(pred_log, measure = "tpr", x.measure = "fpr") #performance measure as ROC
plot(roc.perf_log)

auc.perf_log = performance(pred_log, measure = "auc") #performance measure AUC
auc.perf_log@y.values

opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}
print(opt.cut(roc.perf_log, pred_log)) #formula which finds "optimal" cutoff weighting both sensitivity and specificity equally (TPR and FPR)

acc.perf_log = performance(pred_log, measure = "acc") # find best cutoff according to accuracy
plot(acc.perf_log)

ind_log = which.max( slot(acc.perf_log, "y.values")[[1]] )
acc_log = slot(acc.perf_log, "y.values")[[1]][ind_log]
cutoff_log = slot(acc.perf_log, "x.values")[[1]][ind_log]
print(c(accuracy= acc_log, cutoff = cutoff_log))

pred_log = as.numeric(prob_log > cutoff_log) #ERROR- minimized by maximizing ACCURACY using its cutoff
err_log = mean(pred_log != val_data$target)
err_log

###DECISION TREES### Have to rethink this-> cutoff level and etc
cart = rpart(fmla, data = train_data, method = 'class')
summary(cart)
rpart.plot(cart) # there is no rpart.plot function apparently... plot?

prob_cart = predict(cart, val_data) #prediction

#alternative way using ROCR package
pred_cart = prediction(prob_cart[,2], val_data$target)

roc.perf_cart = performance(pred_cart, measure = "tpr", x.measure = "fpr") #performance measure as ROC
plot(roc.perf_cart)

auc.perf_cart = performance(pred_cart, measure = "auc") #performance measure AUC
auc.perf_cart@y.values

acc.perf_cart = performance(pred_cart, measure = "acc") # find best cutoff according to accuracy
plot(acc.perf_cart)

ind_cart = which.max( slot(acc.perf_cart, "y.values")[[1]] )
acc_cart = slot(acc.perf_cart, "y.values")[[1]][ind_cart]
cutoff_cart = slot(acc.perf_cart, "x.values")[[1]][ind_cart]
print(c(accuracy= acc_cart, cutoff = cutoff_cart))

pred_cart = as.numeric(prob_cart > cutoff_cart) #ERROR- minimized by maximizing ACCURACY using its cutoff
err_cart = mean(pred_cart != val_data$target)
err_cart

## end alternative way
auc_cart = auc(val_data$target, prob_cart[,2]) #auc
auc_cart

plot(roc(val_data$target, prob_cart[,2], direction="<"),
     col="red", lwd=3, main="ROC")

pred_cart = as.numeric(prob_cart[,2]>0.5)
val_data$pred_cart = pred_cart
confusionMatrix(table(val_data$target,val_data$pred_cart), positive = "1")

err_cart = mean(pred_cart != val_data$target) #error
err_cart

### Bagging Decision trees ###
m<-dim(train_data)[1]
ntree<-33
samples<-sapply(1:ntree,
                FUN = function(iter)
                {sample(1:m, size=m, replace=T)}) #replace=T makes it a bootstrap 
head(samples) # inspecting frequencies of samples
hist(samples[,1])

treelist<-lapply(1:ntree, #Training the individual decision trees and return them in a list
                 FUN=function(iter) {
                   samp <- samples[,iter];
                   rpart(fmla,train_data[samp,])
                 }
)
treelist[[1]] #Having a peak at our trees
plot(treelist[[1]])

#predict.bag assumes the underlying classifier returns decision probabilities, not decisions. 
predict.bag<-function(treelist,newdata) {
  preds<-sapply(1:length(treelist),
                FUN=function(iter) {
                  predict(treelist[[iter]],newdata=newdata)
                }
  )
  predsums<-rowSums(preds)
  predsums/length(treelist)
}

prob_bagtree = predict.bag(treelist, newdata=val_data)
pred_bagtree = prediction(prob_bagtree, val_data$target)

roc.perf_bagtree = performance(pred_bagtree, measure = "tpr", x.measure = "fpr") #performance measure as ROC
plot(roc.perf_bagtree)

auc.perf_bagtree = performance(pred_bagtree, measure = "auc") #performance measure AUC
auc.perf_bagtree@y.values

acc.perf_bagtree = performance(pred_bagtree, measure = "acc") # find best cutoff according to accuracy
plot(acc.perf_bagtree)

ind_bagtree = which.max( slot(acc.perf_bagtree, "y.values")[[1]] )
acc_bagtree = slot(acc.perf_bagtree, "y.values")[[1]][ind_bagtree]
cutoff_bagtree = slot(acc.perf_bagtree, "x.values")[[1]][ind_bagtree]
print(c(accuracy= acc_bagtree, cutoff = cutoff_bagtree))

pred_bagtree = as.numeric(prob_bagtree > cutoff_bagtree) #ERROR- minimized by maximizing ACCURACY using its cutoff
err_bagtree = mean(pred_bagtree != val_data$target)
err_bagtree

###NEURAL NETWORKS###
nn = nnet(target~ï..age+sex+cp+trestbps+chol+fbs+restecg+thalach
          +exang+oldpeak+slope+ca+thal, data = train_data, size = 5, decay = 5e-4, maxit = 100)
summary(nn)
##code for optimal cutoffs and etc for logit seems to work here 
prob_nn = predict(nn, val_data)
pred_nn = prediction(prob_nn, val_data$target) #create a prediction object with true values and predicted ones

roc.perf_nn = performance(pred_nn, measure = "tpr", x.measure = "fpr") #performance measure as ROC
plot(roc.perf_nn)

auc.perf_nn = performance(pred_nn, measure = "auc") #performance measure AUC
auc.perf_nn@y.values

print(opt.cut(roc.perf_nn, pred_nn)) #formula which finds "optimal" cutoff weighting both sensitivity and specificity equally (TPR and FPR)

acc.perf_nn = performance(pred_nn, measure = "acc") # find best cutoff according to accuracy
plot(acc.perf_nn)

ind_nn = which.max( slot(acc.perf_nn, "y.values")[[1]] )
acc_nn = slot(acc.perf_nn, "y.values")[[1]][ind_nn]
cutoff_nn = slot(acc.perf_nn, "x.values")[[1]][ind_nn]
print(c(accuracy= acc_nn, cutoff = cutoff_nn))

pred_nn = as.numeric(prob_nn > cutoff_nn) #ERROR- minimized by maximizing ACCURACY using its cutoff
err_nn = mean(pred_nn != val_data$target)
err_nn
## Eriks code for accuracy etc for neural networks
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

pred_xgboost = prediction(prob_xgboost, val_data$target) #create a prediction object with true values and predicted ones

roc.perf_xgboost = performance(pred_xgboost, measure = "tpr", x.measure = "fpr") #performance measure as ROC
plot(roc.perf_xgboost)

auc.perf_xgboost= performance(pred_xgboost, measure = "auc") #performance measure AUC
auc.perf_xgboost@y.values

print(opt.cut(roc.perf_xgboost, pred_xgboost)) #formula which finds "optimal" cutoff weighting both sensitivity and specificity equally (TPR and FPR)

acc.perf_xgboost = performance(pred_xgboost, measure = "acc") # find best cutoff according to accuracy
plot(acc.perf_xgboost)

ind_xgboost = which.max( slot(acc.perf_xgboost, "y.values")[[1]] )
acc_xgboost= slot(acc.perf_xgboost, "y.values")[[1]][ind_xgboost]
cutoff_xgboost = slot(acc.perf_xgboost, "x.values")[[1]][ind_xgboost]
print(c(accuracy= acc_xgboost, cutoff = cutoff_xgboost))

#Eriks code 
auc_xgboost = auc(val_data$target, prob_xgboost) #auc
auc_xgboost

plot(roc(val_data$target, prob_xgboost, direction="<"),
     col="red", lwd=3, main="ROC")

pred_xgboost = as.numeric(prob_xgboost > 0.5)
val_data$pred_xgboost = pred_xgboost
confusionMatrix(table(val_data$target,val_data$pred_xgboost), positive = "1")

err_xgboost = mean(pred_xgboost != val_data$target) #error
err_xgboost

###RANDOM FOREST### have to talk about the ntree and etc settings
rf = randomForest(as.factor(target)~ï..age+sex+cp+trestbps+chol+fbs+restecg+thalach
                  +exang+oldpeak+slope+ca+thal, data = train_data,ntree = 33, nodesize =7, importance = T, proximity = TRUE)

prob_rf = predict(rf, val_data,type='prob') # returns probabilities not 0,1 values with type="prob"

pred_rf = prediction(prob_rf[,2], val_data$target) #create a prediction object with true values and predicted ones

roc.perf_rf = performance(pred_rf, measure = "tpr", x.measure = "fpr") #performance measure as ROC
plot(roc.perf_rf)

auc.perf_rf= performance(pred_rf, measure = "auc") #performance measure AUC
auc.perf_rf@y.values

print(opt.cut(roc.perf_rf, pred_rf)) #formula which finds "optimal" cutoff weighting both sensitivity and specificity equally (TPR and FPR)

acc.perf_rf = performance(pred_rf, measure = "acc") # find best cutoff according to accuracy
plot(acc.perf_rf)

ind_rf = which.max( slot(acc.perf_rf, "y.values")[[1]] )
acc_rf= slot(acc.perf_rf, "y.values")[[1]][ind_rf]
cutoff_rf = slot(acc.perf_rf, "x.values")[[1]][ind_rf]
print(c(accuracy= acc_rf, cutoff = cutoff_rf))
#Eriks code
auc_rf = auc(val_data$target,prob_rf[,2])
auc_rf

plot(roc(val_data$target, prob_rf[,2], direction="<"),
     col="red", lwd=3, main="ROC")

val_data$prob_rf = as.numeric(prob_rf[,2] > 0.5)
confusionMatrix(table(val_data$target,val_data$prob_rf), positive = "1")

err_rf = mean(abs(prob_rf[,2] - val_data$target)) #error of probabilities 
err_rf






####  MESS CODE BELOW  #####
#########################################################################################################################
#########################################################################################################################
###Cross-Validation
k = 100
acc_log = NULL
acc_cart = NULL
acc_nn = NULL
acc_xgboost = NULL
acc_rf = acc_rf
er_log = NULL
er_cart = NULL
er_nn = NULL
er_xgboost = NULL
er_rf = mean(pred_rf != val_data$target)
ac_log = NULL
ac_cart = NULL
ac_nn = NULL
ac_xgboost = NULL
ac_rf = auc.perf_rf@y.values[[1]]
acc.bt.train=NULL
acc.bt.val= NULL
  
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
  #try({
  #  prob_log = predict(logit, val_data) #prediction
  #  pred_log = as.numeric(prob_log > 0.5)
  #  acc_log[i] = confusionMatrix(table(val_data$target,pred_log), positive = "1")$overall['Accuracy']
  #  er_log[i] = mean(pred_log != val_data$target) #error
  #  auc_log[i] = auc(val_data$target,val_data$pred_log)
  #}, TRUE)
  
  ##LOGISTIC - CUTOFF
  prob_log = predict(logit, val_data, type = "response")
  pred_log = prediction(prob_log, val_data$target) #create a prediction object with true values and predicted ones
  
  roc.perf_log = performance(pred_log, measure = "tpr", x.measure = "fpr") #performance measure as ROC
  plot(roc.perf_log)
  
  auc.perf_log = performance(pred_log, measure = "auc") #performance measure AUC
  auc_log[i] = auc.perf_log@y.values[[1]]
  
  print(opt.cut(roc.perf_log, pred_log)) #formula which finds "optimal" cutoff weighting both sensitivity and specificity equally (TPR and FPR)
  
  acc.perf_log = performance(pred_log, measure = "acc") # find best cutoff according to accuracy
  
  ind_log = which.max( slot(acc.perf_log, "y.values")[[1]] )
  acc_log[i] = slot(acc.perf_log, "y.values")[[1]][ind_log]
  cutoff_log = slot(acc.perf_log, "x.values")[[1]][ind_log]
  print(c(accuracy= acc_log, cutoff = cutoff_log))
  
  pred_log = as.numeric(prob_log > cutoff_log) #ERROR- minimized by maximizing ACCURACY using its cutoff
  er_log[i] = mean(pred_log != val_data$target)
  
  ##DECISION TREE
  cart = rpart(fmla, data = train_data, method = 'class')
  #try({
  #  prob_cart = predict(cart, val_data) #prediction
  #  pred_cart = as.numeric(prob_cart[,2] > 0.5)
  #  acc_cart[i] = confusionMatrix(table(val_data$target,pred_cart), positive = "1")$overall['Accuracy']
  #  er_cart[i] = mean(pred_cart != val_data$target) #error
  #  auc_cart[i] = auc(val_data$target,val_data$pred_cart)
  #},TRUE)
  ##DECISION TREE - CUTOFF/BAGGED
  m<-dim(train_data)[1]
  ntree<-33
  samples<-sapply(1:ntree,
                  FUN = function(iter)
                  {sample(1:m, size=m, replace=T)}) #replace=T makes it a bootstrap 
  
  treelist<-lapply(1:ntree, #Training the individual decision trees and return them in a list
                   FUN=function(iter) {
                     samp <- samples[,iter];
                     rpart(fmla,train_data[samp,])
                   }
  )
  #predict.bag assumes the underlying classifier returns decision probabilities, not decisions. 
  predict.bag<-function(treelist,newdata) {
    preds<-sapply(1:length(treelist),
                  FUN=function(iter) {
                    predict(treelist[[iter]],newdata=newdata)
                  }
    )
    predsums<-rowSums(preds)
    predsums/length(treelist)
  }
  
  prob_bagtree = predict.bag(treelist, newdata=val_data)
  pred_bagtree = prediction(prob_bagtree, val_data$target)
  
  roc.perf_bagtree = performance(pred_bagtree, measure = "tpr", x.measure = "fpr") #performance measure as ROC

  auc.perf_bagtree = performance(pred_bagtree, measure = "auc") #performance measure AUC
  auc_cart[i] = auc.perf_bagtree@y.values[[1]]
  
  acc.perf_bagtree = performance(pred_bagtree, measure = "acc") # find best cutoff according to accuracy
  
  ind_bagtree = which.max( slot(acc.perf_bagtree, "y.values")[[1]] )
  acc_cart[i] = slot(acc.perf_bagtree, "y.values")[[1]][ind_bagtree]
  cutoff_bagtree = slot(acc.perf_bagtree, "x.values")[[1]][ind_bagtree]
  print(c(accuracy= acc_bagtree, cutoff = cutoff_bagtree))
  
  pred_bagtree = as.numeric(prob_bagtree > cutoff_bagtree) #ERROR- minimized by maximizing ACCURACY using its cutoff
  er_cart[i] = mean(pred_bagtree != val_data$target)
  
  ##NEURAL NET
  nn = nnet(target~ï..age+sex+cp+trestbps+chol+fbs+restecg+thalach
            +exang+oldpeak+slope+ca+thal, data = train_data, size = 5, decay = 5e-4, maxit = 100)
  #try({
  #  prob_nn = predict(nn, val_data)
  #  pred_nn = as.numeric(prob_nn > 0.5)
  #  acc_nn[i] = confusionMatrix(table(val_data$target,pred_nn), positive = "1")$overall['Accuracy']
  #  er_nn[i] = mean(pred_nn != val_data$target) #error
  #  auc_nn[i] = auc(val_data$target,val_data$pred_nn)
  #},TRUE)
  ##NEURAL NET - CUTOFF
  prob_nn = predict(nn, val_data)
  pred_nn = prediction(prob_nn, val_data$target) #create a prediction object with true values and predicted ones
  
  roc.perf_nn = performance(pred_nn, measure = "tpr", x.measure = "fpr") #performance measure as ROC
  
  auc.perf_nn = performance(pred_nn, measure = "auc") #performance measure AUC
  auc_nn[i] = auc.perf_nn@y.values[[1]]
  
  print(opt.cut(roc.perf_nn, pred_nn)) #formula which finds "optimal" cutoff weighting both sensitivity and specificity equally (TPR and FPR)
  
  acc.perf_nn = performance(pred_nn, measure = "acc") # find best cutoff according to accuracy
  
  ind_nn = which.max( slot(acc.perf_nn, "y.values")[[1]] )
  acc_nn[i] = slot(acc.perf_nn, "y.values")[[1]][ind_nn]
  cutoff_nn = slot(acc.perf_nn, "x.values")[[1]][ind_nn]
  print(c(accuracy= acc_nn, cutoff = cutoff_nn))
  
  pred_nn = as.numeric(prob_nn > cutoff_nn) #ERROR- minimized by maximizing ACCURACY using its cutoff
  er_nn[i] = mean(pred_nn != val_data$target)
  
  ##BOOSTED TREE
  sparse_matrix = sparse.model.matrix(target ~ ., data = train_data)[,-1] #dummy contrast coding of categorical variables to fit the xgboost
  sparse_matrix_val = sparse.model.matrix(target ~ ., data = val_data[,colnames(train_data)])[,-1]
    
  labels = train_data$target
    
  bst = xgboost(data = sparse_matrix, label = labels, max_depth = 4,
                  eta = 1, nthread = 2, nrounds = 10,objective = "binary:logistic")
  #try({
  #  prob_xgboost = predict(bst, sparse_matrix_val)
  #  pred_xgboost = as.numeric(prob_xgboost > 0.5)
  #  acc_xgboost[i] = confusionMatrix(table(val_data$target,pred_xgboost), positive = "1")$overall['Accuracy']
  #  er_xgboost[i] = mean(pred_xgboost != val_data$target) #error
  #  auc_xgboost[i] = auc(val_data$target,val_data$pred_xgboost)
  #}, TRUE)
  ##XGBOOST - CUTOFF
  importance = xgb.importance(feature_names = colnames(sparse_matrix), model = bst) #importance of features
  head(importance)
  
  prob_xgboost = predict(bst, sparse_matrix_val)
  
  pred_xgboost = prediction(prob_xgboost, val_data$target) #create a prediction object with true values and predicted ones
  
  roc.perf_xgboost = performance(pred_xgboost, measure = "tpr", x.measure = "fpr") #performance measure as ROC
  
  auc.perf_xgboost= performance(pred_xgboost, measure = "auc") #performance measure AUC
  auc_xgboost[i] = auc.perf_xgboost@y.values
  
  print(opt.cut(roc.perf_xgboost, pred_xgboost)) #formula which finds "optimal" cutoff weighting both sensitivity and specificity equally (TPR and FPR)
  
  acc.perf_xgboost = performance(pred_xgboost, measure = "acc") # find best cutoff according to accuracy
  plot(acc.perf_xgboost)
  
  ind_xgboost = which.max( slot(acc.perf_xgboost, "y.values")[[1]] )
  acc_xgboost[i] = slot(acc.perf_xgboost, "y.values")[[1]][ind_xgboost]
  cutoff_xgboost = slot(acc.perf_xgboost, "x.values")[[1]][ind_xgboost]
  print(c(accuracy= acc_xgboost, cutoff = cutoff_xgboost))
  
  pred_xgboost = as.numeric(prob_xgboost > cutoff_xgboost) #ERROR- minimized by maximizing ACCURACY using its cutoff
  er_xgboost[i] = mean(pred_xgboost != val_data$target)
  
  ## Bagged Tree
  try({
    samples<-sapply(1:ntree,
                    FUN = function(iter)
                    {sample(1:m, size=m, replace=T)})
    treelist<-lapply(1:ntree, #Training the individual decision trees and return them in a list
                     FUN=function(iter) {
                       samp <- samples[,iter];
                       rpart(fmla,train_data[samp,])
                     }
    )
    predict.bag<-function(treelist,newdata) {
      preds<-sapply(1:length(treelist),
                    FUN=function(iter) {
                      predict(treelist[[iter]],newdata=newdata)
                    }
      )
      predsums<-rowSums(preds)
      predsums/length(treelist)
    }
    acc.bt.train[[i]] =accuracyMeasures(predict.bag(treelist, newdata=train_data),
                     train_data$target==1,
                     name="bagging, training")
    acc.bt.val[[i]] = accuracyMeasures(predict.bag(treelist, newdata=val_data),
                     val_data$target==1,
                     name="bagging, test")
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
acc_rf = acc_rf
er_log = NULL
er_cart = NULL
er_nn = NULL
er_xgboost = NULL
er_rf = mean(pred_rf != val_data$target)
ac_log = NULL
ac_cart = NULL
ac_nn = NULL
ac_xgboost = NULL
ac_rf = auc.perf_rf@y.values[[1]]
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

sp_sen_logit= opt.cut(many.roc.perf, manypred) # find optimal cutoffs for each estimation according to trade off sensitivity and specitivity
av_sp_sen_logit = c(mean(sp_sen_logit[1,]), mean(sp_sen_logit[2,]), mean(sp_sen_logit[3,]))

many.acc.perf = performance(manypred, measure = "acc") # find cutoff for highest possible accuracy
sapply(manypred@labels, function(x) mean(x == 1))

logit_acc = mapply(function(x, y){
            ind = which.max( y )
            acc = y[ind]
            cutoff = x[ind]
            return(c(accuracy= acc, cutoff = cutoff))
            }, slot(many.acc.perf, "x.values"), slot(many.acc.perf, "y.values"))

av_logit_acc= c(mean(logit_acc[1,]), mean(logit_acc[2,])) #average highest accuracy and corresponding average cutoff

manypauc.perf = performance(manypred, measure = "auc") # area under the curve values for all estimations
logit_auc_mean = mean(unlist(manypauc.perf@y.values))

#error average here but must be still worked on (maybe not best solution like this)
pred_log = vector(mode = "list", length = 84)
err_log= vector(mode = "list", length = 84)
logit_acc_list =as.list(logit_acc[1,])

for (j in 1:84){
  pred_log[[j]] = as.numeric(prob_log[[j]] > logit_acc_list[[j]])
  err_log[[j]] = mean(pred_log[[j]] != logit_pred[[j]])
  j=j+1
}
av_error_log= mean(unlist(err_log))

################
###########TRIES
sparse_matrix = sparse.model.matrix(target ~ ., data = train_data)[,-1] #dummy contrast coding of categorical variables to fit the xgboost
sparse_matrix_val = sparse.model.matrix(target ~ ., data = val_data[,colnames(train_data)])[,-1]

labels = as.numeric(train_data$target)
ts_label = as.numeric(val_data$target)

dtrain = xgb.DMatrix(data=sparse_matrix, label = labels)
dval = xgb.DMatrix(data = sparse_matrix_val, label = ts_label)

params = list(booster = "gbtree", objective = "binary:logistic",
              eta = 0.3, gamma = 0, max_depth = 6, min_child_weight = 1,
              subsample = 1, colsample_bytree = 1)

xgbcv = xgb.cv(params = params, data = dtrain, nrounds = 100, nfold = 5, #11 lowest
         showsd = 5, stratified = T, print_every_n = 1,
         early_stoppping_rounds = 20, maximize = F)

min(xgbcv$evaluation_log$test_error_mean)

fact_col = colnames(train_data)[sapply(train_data,is.character)]
for(i in fact_col) set (train_data, j = i, value = factor(data_train[i]))
for(i in fact_col) set (val_data, j = i, value = factor(data_train[i]))

traintask = makeClassifTask(data = train_data, target = "target")
testtask = makeClassifTask(data = val_data, target = "target")

traintask = createDummyFeatures(obj = traintask)
testtask = createDummyFeatures(obj = testtask)

lrn = makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals = list(objective = "binary:logistic", eval_metric = "error",
                    nrounds = 11L, eta = 0.1)

params = makeParamSet(makeDiscreteParam("booster", 
                                        values = c("gbtree,gblinear")),
                      makeIntegerParam("max_depth", lower = 3L, upper = 10L),
                      makeNumericParam("min_child_weight", lower = 1L, upper = 10L),
                      makeNumericParam("subsample", lower = 0.5, upper = 1),
                      makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
                      makeNumericParam("gamma",lower = 0, upper = 2))
rdesc = makeResampleDesc("CV", stratify = T, iters = 5L)
ctrl = makeTuneControlRandom(maxit = 10L)

parallelStartSocket(cpus=detectCores())
mytune = tuneParams(learner = lrn, task = traintask, resampling = rdesc,
                    measures = acc, par.set = params, control = ctrl,
                    show.info = T)
mytune$y
