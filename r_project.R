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
data$target = as.factor(data$target) #outcome variable (heart disease or not)

data = data[data$thal!=0,]  #removing and merging very rare levels /not informative
data$thal = factor(data$thal)
#data$ca[data$ca==4] = 3
data$ca = factor(data$ca)

str(data) #variables
summary(data) #variable statistics
sapply(data, sd)
apply(data, 2, function(x) {sum(is.na(x))}) #NAs inspection, 0 present
summary(data)

data_cv = data #cross-validation

summary(data_cv$target)

n = nrow(data) #shuffling the data
set.seed(104)
data = data[sample(n),] 

set.seed(46)
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
er_log = NULL
auc_log = NULL
cut_log = NULL

for (i in 1:k){ #sample randomly 100 times
  n = nrow(data_cv)
  set.seed(21)
  data_temp = data_cv[sample(n),] 
  rand_rows = sample(1:nrow(data_temp), 0.7*nrow(data_temp)) #split into training and validation dataset
  train_data = data_temp[rand_rows, ]
  val_data = data_temp[-rand_rows, ]

  ##LOGISTISTIC ##DONE
  logit = glm(fmla, data = train_data, family = "binomial")
  try({
    prob_log = predict(logit, val_data, type = "response")
    pred_log = ROCR::prediction(prob_log, val_data$target) #create a prediction object with true values and predicted ones
    
    roc.perf_log = ROCR::performance(pred_log, measure = "tpr", x.measure = "fpr") #performance measure as ROC
    
    auc.perf_log = ROCR::performance(pred_log, measure = "auc") #performance measure AUC
    auc_log[i] = auc.perf_log@y.values[[1]]
    
    acc.perf_log = ROCR::performance(pred_log, measure = "acc") # find best cutoff according to accuracy
    
    ind_log = which.max( slot(acc.perf_log, "y.values")[[1]] )
    acc_log[i] = slot(acc.perf_log, "y.values")[[1]][ind_log]
    cutoff_log = slot(acc.perf_log, "x.values")[[1]][ind_log]
    cut_log[i] = cutoff_log
    
    pred_log = as.numeric(prob_log > cutoff_log) #ERROR- minimized by maximizing ACCURACY using its cutoff
    er_log[i] = mean(pred_log != val_data$target)
  },TRUE)
}

##DECISION TREE - BAGGED ##DONE
m<-dim(train_data)[1]
ntree<-500
set.seed(65)
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
  
  roc.perf_bagtree = ROCR::performance(pred_bagtree, measure = "tpr", x.measure = "fpr") #performance measure as ROC
  
  auc.perf_bagtree = ROCR::performance(pred_bagtree, measure = "auc") #performance measure AUC
  auc_cart = auc.perf_bagtree@y.values[[1]]
  
  acc.perf_bagtree = ROCR::performance(pred_bagtree, measure = "acc") # find best cutoff according to accuracy
  
  ind_bagtree = which.max( slot(acc.perf_bagtree, "y.values")[[1]] )
  acc_cart = slot(acc.perf_bagtree, "y.values")[[1]][ind_bagtree]
  cutoff_bagtree = slot(acc.perf_bagtree, "x.values")[[1]][ind_bagtree]
  cut_cart = cutoff_bagtree
  
  pred_bagtree = as.numeric(prob_bagtree > cutoff_bagtree) #ERROR- minimized by maximizing ACCURACY using its cutoff
  er_cart = mean(pred_bagtree != val_data$target)
},TRUE)

#er_cart = mean(abs(prob_bagtree - val_data$target)) - probability error 

###RANDOM FOREST ##DONE
set.seed(87)
rf = randomForest(as.factor(target)~ï..age+sex+cp+trestbps+chol+fbs+restecg+thalach
                  +exang+oldpeak+slope+ca+thal, nodesize = 1 ,data = train_data, proximity = TRUE)

prob_rf = predict(rf, val_data,type='prob') # returns probabilities not 0,1 values with type="prob"

pred_rf = ROCR::prediction(prob_rf[,2], val_data$target) #create a prediction object with true values and predicted ones

roc.perf_rf = ROCR::performance(pred_rf, measure = "tpr", x.measure = "fpr") #performance measure as ROC
plot(roc.perf_rf)

auc.perf_rf= ROCR::performance(pred_rf, measure = "auc") #performance measure AUC
auc_rf = auc.perf_rf@y.values[[1]]

print(opt.cut(roc.perf_rf, pred_rf)) #formula which finds "optimal" cutoff weighting both sensitivity and specificity equally (TPR and FPR)

acc.perf_rf = ROCR::performance(pred_rf, measure = "acc") # find best cutoff according to accuracy
plot(acc.perf_rf)

ind_rf = which.max( slot(acc.perf_rf, "y.values")[[1]] )
acc_rf= slot(acc.perf_rf, "y.values")[[1]][ind_rf]
cutoff_rf = slot(acc.perf_rf, "x.values")[[1]][ind_rf]
cut_rf = cutoff_rf
print(c(accuracy= acc_rf, cutoff = cutoff_rf))

pred_rf = as.numeric(prob_rf[,2] > cutoff_rf) #ERROR- minimized by maximizing ACCURACY using its cutoff
er_rf = mean(pred_rf != val_data$target)


###XGBOOST### ##DONE
sparse_matrix = sparse.model.matrix(target ~ ., data = train_data)[,-1] #dummy contrast coding of categorical variables to fit the xgboost
sparse_matrix_val = sparse.model.matrix(target ~ ., data = val_data[,colnames(train_data)])[,-1]

labels = as.numeric(train_data$target)
ts_label = as.numeric(val_data$target)

fact_col = colnames(train_data)[sapply(train_data,is.character)]
for(i in fact_col) set (train_data, j = i, value = factor(data_train[i]))
for(i in fact_col) set (val_data, j = i, value = factor(data_train[i]))

traintask = makeClassifTask(data = train_data, target = "target")
testtask = makeClassifTask(data = val_data, target = "target")

traintask = createDummyFeatures(obj = traintask)
testtask = createDummyFeatures(obj = testtask)

lrn = makeLearner("classif.xgboost",predict.type = "prob")
lrn$par.vals = list(objective = "binary:logistic", eval_metric = "error")

params = makeParamSet(makeDiscreteParam("booster", 
                                        values = c("gbtree","gblinear")),
                      makeIntegerParam("max_depth", lower = 3L, upper = 10L),
                      makeIntegerParam("nrounds", lower = 5L, upper = 30L),
                      makeNumericParam("min_child_weight", lower = 1L, upper = 10L),
                      makeNumericParam("subsample", lower = 0.5, upper = 1),
                      makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
                      makeNumericParam("gamma",lower = 0, upper = 2),
                      makeNumericParam("eta", lower = 0.1, upper = 3))
rdesc = makeResampleDesc("CV", stratify = T, iters = 5L)
ctrl = makeTuneControlRandom(maxit = 100L)
#ctrl = makeTuneControlGrid(resolution = c(max_depth = 10, nrounds = 10, min_child_weight = 50,
#                                          subsample = 50, colsample_bytree = 50, gamma = 50,
#                                          eta = 50), tune.threshold = FALSE)

parallelStartSocket(cpus=detectCores())
set.seed(121)
mytune = tuneParams(learner = lrn, task = traintask, resampling = rdesc,
                    measures = acc, par.set = params, control = ctrl,
                    show.info = F)

lrn_tune = setHyperPars(lrn, par.vals = mytune$x)
set.seed(11)
xgmodel = train(learner = lrn_tune, task = traintask)
xgpred = predict(xgmodel,testtask)
acc_xgboost = confusionMatrix(xgpred$data$response,xgpred$data$truth)$overall[1]
er_xgboost = mean(xgpred$data$response != xgpred$data$truth)

auc = generateThreshVsPerfData(xgpred, measures = list(fpr, tpr, mmce))
plotROCCurves(auc)
auc_xgboost = mlr::performance(xgpred, mlr::auc)


##NNET
traintask = makeClassifTask(data = train_data, target = "target")
testtask = makeClassifTask(data = val_data, target = "target")

traintask = createDummyFeatures(obj = traintask)
testtask = createDummyFeatures(obj = testtask)

lrn = makeLearner("classif.nnet",predict.type = "prob")
lrn$par.vals = list(maxit = 100, na.action = na.omit)

params = makeParamSet(
                      makeIntegerParam("size", lower = 2L, upper = 10L),
                      makeNumericParam("decay", lower = -3, upper = 3))
rdesc = makeResampleDesc("CV", stratify = T, iters = 5L)
#design = generateGridDesign(params, resolution = c(size = 10, decay = 100))
ctrl = makeTuneControlRandom(maxit = 100L)
#ctrl = makeTuneControlGrid(resolution = c(size = 10, decay = 500), tune.threshold = FALSE)
set.seed(49)
mytune = tuneParams(learner = lrn, task = traintask, resampling = rdesc,
                    measures = acc, par.set = params, control = ctrl,
                    show.info = F)
mytune$y

lrn_tune = setHyperPars(lrn, par.vals = mytune$x)
set.seed(95)
nnetmodel = train(learner = lrn_tune, task = traintask)
nnetpred = predict(nnetmodel,testtask)
acc_nn = confusionMatrix(nnetpred$data$response,nnetpred$data$truth)$overall[1]
er_nn = mean(nnetpred$data$response != nnetpred$data$truth)

auc = generateThreshVsPerfData(nnetpred, measures = list(fpr, tpr, mmce))
plotROCCurves(auc)
auc_nn = mlr::performance(nnetpred, mlr::auc)


#######################
####Replicated Paper###
#######################

###LOGISTIC REGRESSION### ##DONE
logit = glm(fmla, data = train_data, family = "binomial")

prob_log = predict(logit, val_data, type = "response")
pred_log = ROCR::prediction(prob_log, val_data$target) #create a prediction object with true values and predicted ones

roc.perf_log = ROCR::performance(pred_log, measure = "tpr", x.measure = "fpr") #performance measure as ROC
plot(roc.perf_log)

auc.perf_log = ROCR::performance(pred_log, measure = "auc") #performance measure AUC
auc_log_old = auc.perf_log@y.values[[1]]

opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}
print(opt.cut(roc.perf_log, pred_log)) #formula which finds "optimal" cutoff weighting both sensitivity and specificity equally (TPR and FPR)

acc.perf_log = ROCR::performance(pred_log, measure = "acc") # find best cutoff according to accuracy
plot(acc.perf_log)

ind_log = which.max( slot(acc.perf_log, "y.values")[[1]] )
acc_log_old = slot(acc.perf_log, "y.values")[[1]][ind_log]
cutoff_log = slot(acc.perf_log, "x.values")[[1]][ind_log]
print(c(accuracy= acc_log_old, cutoff = cutoff_log))

pred_log = as.numeric(prob_log > cutoff_log) #ERROR- minimized by maximizing ACCURACY using its cutoff
er_log_old = mean(pred_log != val_data$target)
er_log_old

###DECISION TREES### ##DONE
cart = rpart(fmla, data = train_data, method = 'class')
rpart.plot(cart)

prob_cart = predict(cart, val_data) #prediction

pred_cart = ROCR::prediction(prob_cart[,2], val_data$target)

roc.perf_cart = ROCR::performance(pred_cart, measure = "tpr", x.measure = "fpr") #performance measure as ROC
plot(roc.perf_cart)

auc.perf_cart = ROCR::performance(pred_cart, measure = "auc") #performance measure AUC
auc_cart_old = auc.perf_cart@y.values[[1]]
print(opt.cut(roc.perf_cart, pred_cart))

acc.perf_cart = ROCR::performance(pred_cart, measure = "acc") # find best cutoff according to accuracy
plot(acc.perf_cart)

ind_cart = which.max( slot(acc.perf_cart, "y.values")[[1]] )
acc_cart_old = slot(acc.perf_cart, "y.values")[[1]][ind_cart]
cutoff_cart = slot(acc.perf_cart, "x.values")[[1]][ind_cart]
print(c(accuracy= acc_cart_old, cutoff = cutoff_cart))

pred_cart = as.numeric(prob_cart[,2] > cutoff_cart) #ERROR- minimized by maximizing ACCURACY using its cutoff
er_cart_old = mean(pred_cart != val_data$target)
er_cart_old

###NEURAL NETWORKS### ##DONE
set.seed(69)
nn = nnet(target~ï..age+sex+cp+trestbps+chol+fbs+restecg+thalach
          +exang+oldpeak+slope+ca+thal, data = train_data, size = 5, decay = 5e-4, maxit = 100)

prob_nn = predict(nn, val_data)
pred_nn = ROCR::prediction(prob_nn, val_data$target) #create a prediction object with true values and predicted ones

roc.perf_nn = ROCR::performance(pred_nn, measure = "tpr", x.measure = "fpr") #performance measure as ROC
plot(roc.perf_nn)

auc.perf_nn = ROCR::performance(pred_nn, measure = "auc") #performance measure AUC
auc_nn_old = auc.perf_nn@y.values[[1]]

print(opt.cut(roc.perf_nn, pred_nn)) #formula which finds "optimal" cutoff weighting both sensitivity and specificity equally (TPR and FPR)

acc.perf_nn = ROCR::performance(pred_nn, measure = "acc") # find best cutoff according to accuracy
plot(acc.perf_nn)

ind_nn = which.max( slot(acc.perf_nn, "y.values")[[1]] )
acc_nn_old = slot(acc.perf_nn, "y.values")[[1]][ind_nn]
cutoff_nn = slot(acc.perf_nn, "x.values")[[1]][ind_nn]
print(c(accuracy= acc_nn_old, cutoff = cutoff_nn))

pred_nn = as.numeric(prob_nn > cutoff_nn) #ERROR- minimized by maximizing ACCURACY using its cutoff
er_nn_old = mean(pred_nn != val_data$target)
er_nn_old

################
###STATISTICS###
################

###NEW###
print(mean(acc_log, na.rm = TRUE))
print(acc_cart)
print(acc_nn)
print(acc_xgboost)
print(acc_rf)

print(mean(er_log, na.rm = TRUE))
print(er_cart)
print(er_nn)
print(er_xgboost)
print(er_rf)

print(mean(auc_log, na.rm = TRUE))
print(auc_cart)
print(auc_nn)
print(auc_xgboost)
print(auc_rf)

print(mean(cut_log, na.rm = TRUE))
print(cut_cart)
print(cut_rf)

###OLD###
print(acc_log_old)
print(acc_cart_old)
print(acc_nn_old)

print(er_log_old)
print(er_cart_old)
print(er_nn_old)

print(auc_log_old)
print(auc_cart_old)
print(auc_nn_old)
