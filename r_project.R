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
data$ca[data$ca==4] = 3
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

set.seed(45)
rand_rows = sample(1:nrow(data), 0.6*nrow(data)) #split into training and validation dataset
train_data = data[rand_rows, ]
val_data = data[-rand_rows, ]

vars = colnames(data)[-length(colnames(data))] #creating a formula
y = "target" 
fmla = paste(y, paste(vars, collapse="+"),sep="~")
fmla

#### creating 2 functions which we will use ###
opt.cut = function(perf, pred){ # function which finds optimal cutoff level maximizing a trade-off between Sensitivity& Specifivity
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}
performanceMeasures<-function(prob, truth, name="model") { #Function for various accuracy measures we use.
  pred= ROCR::prediction(prob, truth)
  roc = ROCR::performance(pred, measure = "tpr", x.measure = "fpr") #performance measure as ROC
  auc.perf= ROCR::performance(pred, measure = "auc") #performance measure AUC
  auc= auc.perf@y.values[[1]]
  sen_spe = opt.cut(roc, pred)
  acc.perf = ROCR::performance(pred, measure = "acc") # find best cutoff according to accuracy
  ind = which.max( slot(acc.perf, "y.values")[[1]] )
  acc = slot(acc.perf, "y.values")[[1]][ind]
  cutoff= slot(acc.perf, "x.values")[[1]][ind]
  pred_cut = as.numeric(prob > cutoff) #ERROR- minimized by maximizing ACCURACY using its cutoff
  er = mean(pred_cut != truth)
  data.frame(model=name, Area_under_the_curve = auc, Sensitivity = sen_spe[1], 
             Specificity= sen_spe[2], Cutoff_Sen_Spe= sen_spe[3], Accuracy=acc, Cutoff_acc = cutoff, error=er)
}
#######################
####Replicated Paper###
#######################

###LOGISTIC REGRESSION### ##DONE
logit = glm(fmla, data = train_data, family = "binomial")

prob_log_train = predict(logit, train_data, type = "response")
prob_log_test = predict(logit, val_data, type = "response")

perf_log_train = performanceMeasures(prob_log_train, train_data$target, name="Logit Training")
perf_log_train
perf_log_test = performanceMeasures(prob_log_test, val_data$target, name="Logit Testing")

###DECISION TREES### ##DONE
cart = rpart(fmla, data = train_data, method = 'class')
rpart.plot(cart)

prob_cart_train = predict(cart, train_data)
prob_cart_test = predict(cart, val_data) #prediction

perf_cart_train = performanceMeasures(prob_cart_train[,2], train_data$target, name="DECISION TREES Training")
perf_cart_train
perf_cart_test = performanceMeasures(prob_cart_test[,2], val_data$target, name="DECISION TREES Testing")
perf_cart_test

###NEURAL NETWORKS### ##DONE
set.seed(69)
nn = nnet(target~ï..age+sex+cp+trestbps+chol+fbs+restecg+thalach
          +exang+oldpeak+slope+ca+thal, data = train_data, size = 5, decay = 5e-4, maxit = 100)

prob_nn_train = predict(nn, train_data)
prob_nn_test = predict(nn, val_data) #prediction

perf_nn_train = performanceMeasures(prob_nn_train, train_data$target, name="NEURAL NETWORKS Training")
perf_nn_train
perf_nn_test = performanceMeasures(prob_nn_test, val_data$target, name="NEURAL NETWORKS Testing")
perf_nn_test

###########################
#### Additional methods####
###########################

##DECISION TREE - BAGGED ##DONE 
m<-dim(train_data)[1]
ntree<-500
set.seed(65)
samples<-sapply(1:ntree,
                FUN = function(iter)
                {sample(1:m, size=m, replace=T)}) #replace=T makes it a bootstrap 

treelist<-lapply(1:ntree, #Training the individual decision trees and return them in a list
                 FUN=function(iter) {
                   samp <- samples[,iter];
                   rpart(fmla,train_data[samp,], method= "class")
                 }
)
#predict.bag assumes the underlying classifier returns decision probabilities, not decisions. 
predict.bag<-function(treelist,newdata) {
  preds<-lapply(1:length(treelist),
                FUN=function(iter) {
                  predict(treelist[[iter]],newdata=newdata)
                }
  )
  preds
}
prob_bagtree_train = predict.bag(treelist, newdata=train_data)
prob_bagtree_train= as.data.frame(prob_bagtree_train)
prob_bagtree_train_1= rowSums(prob_bagtree_train[, seq(2, ncol(prob_bagtree_train), 2)])/length(treelist)

prob_bagtree_test = predict.bag(treelist, newdata=val_data)
prob_bagtree_test= as.data.frame(prob_bagtree_test)
prob_bagtree_test_1= rowSums(prob_bagtree_test[, seq(2, ncol(prob_bagtree_test), 2)])/length(treelist)

perf_bagtree_train = performanceMeasures(prob_bagtree_train_1, train_data$target, name="Bagged Trees Training")
perf_bagtree_train
perf_bagtree_test = performanceMeasures(prob_bagtree_test_1, val_data$target, name="Bagged Trees Testing")
perf_bagtree_test

###RANDOM FOREST ##DONE
set.seed(87)
rf = randomForest(as.factor(target)~ï..age+sex+cp+trestbps+chol+fbs+restecg+thalach
                  +exang+oldpeak+slope+ca+thal, nodesize = 1 ,data = train_data, proximity = TRUE)

prob_rf_train = predict(rf, train_data, type='prob') # returns probabilities not 0,1 values with type="prob"
prob_rf_test = predict(rf, val_data, type='prob')

perf_rf_train = performanceMeasures(prob_rf_train[,2], train_data$target, name="RANDOM FOREST Training")
perf_rf_train
perf_rf_test = performanceMeasures(prob_rf_test[,2], val_data$target, name="RANDOM FOREST Testing")
perf_rf_test

######################
####Cross-validated###
######################

k = 100
prob_log_train =  NULL
prob_log_test = NULL
perf_log_train= as.data.frame( matrix(0, ncol = 0, nrow = 0))
perf_log_test=as.data.frame( matrix(0, ncol = 0, nrow = 0))
prob_cart_train= NULL
prob_cart_test= NULL
perf_cart_train= as.data.frame( matrix(0, ncol = 0, nrow = 0))
perf_cart_test=as.data.frame( matrix(0, ncol = 0, nrow = 0))
prob_nn_train= NULL
prob_nn_test= NULL
perf_nn_train= as.data.frame( matrix(0, ncol = 0, nrow = 0))
perf_nn_test=as.data.frame( matrix(0, ncol = 0, nrow = 0))
prob_rf_train= NULL
prob_rf_test= NULL
perf_rf_train= as.data.frame( matrix(0, ncol = 0, nrow = 0))
perf_rf_test=as.data.frame( matrix(0, ncol = 0, nrow = 0))
prob_bagtree_train= NULL
prob_bagtree_test= NULL
perf_bagtree_train= as.data.frame( matrix(0, ncol = 0, nrow = 0))
perf_bagtree_test=as.data.frame( matrix(0, ncol = 0, nrow = 0))

for (i in 1:k){ #sample randomly 100 times
  n = nrow(data_cv)
  data_temp = data_cv[sample(n),] 
  rand_rows = sample(1:nrow(data_temp), 0.7*nrow(data_temp)) #split into training and validation dataset
  train_data = data_temp[rand_rows, ]
  val_data = data_temp[-rand_rows, ]
  
  ##LOGISTISTIC ##DONE
  logit = glm(fmla, data = train_data, family = "binomial")
  try({
    prob_log_train = predict(logit, train_data, type = "response")
    prob_log_test = predict(logit, val_data, type = "response")
    perf_log_train = rbind(perf_log_train, performanceMeasures(prob_log_train, train_data$target, name="Logit Training"))
    perf_log_test = rbind(perf_log_test,performanceMeasures(prob_log_test, val_data$target, name="Logit Testing"))
  },TRUE)
  cart = rpart(fmla, data = train_data, method = 'class')
  try({
    prob_cart_train = predict(cart, train_data)
    prob_cart_test = predict(cart, val_data) #prediction
    perf_cart_train = rbind(perf_cart_train,performanceMeasures(prob_cart_train[,2], train_data$target, name="DECISION TREES Training"))
    perf_cart_test =rbind(perf_cart_test, performanceMeasures(prob_cart_test[,2], val_data$target, name="DECISION TREES Testing"))
    },TRUE)
  nn = nnet(target~ï..age+sex+cp+trestbps+chol+fbs+restecg+thalach
            +exang+oldpeak+slope+ca+thal, data = train_data, size = 5, decay = 5e-4, maxit = 100)
  try({
    prob_nn_train = predict(nn, train_data)
    prob_nn_test = predict(nn, val_data) #prediction
    perf_nn_train =rbind(perf_nn_train, performanceMeasures(prob_nn_train, train_data$target, name="NEURAL NETWORKS Training"))
    perf_nn_test = rbind(perf_nn_test, performanceMeasures(prob_nn_test, val_data$target, name="NEURAL NETWORKS Testing"))
    },TRUE)
  rf = randomForest(as.factor(target)~ï..age+sex+cp+trestbps+chol+fbs+restecg+thalach
                    +exang+oldpeak+slope+ca+thal, nodesize = 1 ,data = train_data, proximity = TRUE)
  try({
    prob_rf_train = predict(rf, train_data, type='prob') # returns probabilities not 0,1 values with type="prob"
    prob_rf_test = predict(rf, val_data, type='prob')
    perf_rf_train = rbind(perf_rf_train,performanceMeasures(prob_rf_train[,2], train_data$target, name="RANDOM FOREST Training"))
    perf_rf_test = rbind(perf_rf_test,performanceMeasures(prob_rf_test[,2], val_data$target, name="RANDOM FOREST Testing"))
    },TRUE)
  
  samples<-sapply(1:ntree,
                  FUN = function(iter)
                  {sample(1:m, size=m, replace=T)}) #replace=T makes it a bootstrap 
  treelist<-lapply(1:ntree, #Training the individual decision trees and return them in a list
                     FUN=function(iter) {
                       samp <- samples[,iter];
                       rpart(fmla,train_data[samp,], method= "class")
                     }
    )
  try({
    prob_bagtree_train = predict.bag(treelist, newdata=train_data)
    prob_bagtree_train= as.data.frame(prob_bagtree_train)
    prob_bagtree_train_1= rowSums(prob_bagtree_train[, seq(2, ncol(prob_bagtree_train), 2)])/length(treelist)
    
    prob_bagtree_test = predict.bag(treelist, newdata=val_data)
    prob_bagtree_test= as.data.frame(prob_bagtree_test)
    prob_bagtree_test_1= rowSums(prob_bagtree_test[, seq(2, ncol(prob_bagtree_test), 2)])/length(treelist)
    
    perf_bagtree_train = rbind(perf_bagtree_train, performanceMeasures(prob_bagtree_train_1, train_data$target, name="Bagged Trees Training"))
    perf_bagtree_test = rbind(perf_bagtree_test, performanceMeasures(prob_bagtree_test_1, val_data$target, name="Bagged Trees Testing"))
    },TRUE)
}

summary(perf_log_train)
summary(perf_log_test)

summary(perf_cart_train)
summary(perf_cart_test)

summary(perf_nn_train)
summary(perf_nn_test)

summary(perf_rf_train)
summary(perf_rf_test)

summary(perf_bagtree_train)
summary(perf_bagtree_test)


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
