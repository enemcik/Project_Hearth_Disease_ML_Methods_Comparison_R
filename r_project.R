install.packages(c("parallel","parallelMap", "mlr"))
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

data = data[data$thal!=0,]  #removing and merging very rare levels /not informative
data$ca[data$ca==4] = 3
data$restecg[data$restecg==2] = 1 
names(data)[names(data)=="ï..age"]="age" # had a weird name "i with double points" which when the file is downloaded is changed to "?..age"

data$sex = as.factor(data$sex) #gender
data$cp = as.factor(data$cp) #chest pain
data$fbs = as.factor(data$fbs) #fasting blood sugar
data$restecg = as.factor(data$restecg) #resting ECG
data$exang = as.factor(data$exang) # induced angina
data$slope = as.factor(data$slope) #slope
data$ca = as.factor(data$ca) #number of colored vessels
data$thal = as.factor(data$thal) #thal
data$target = as.factor(data$target) #outcome variable (heart disease or not)


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
perf_log_test
###DECISION TREES### ##DONE
cart = rpart(fmla, data = train_data, method = 'class')
rpart.plot(cart)

prob_cart_train = predict(cart, train_data)
prob_cart_test = predict(cart, val_data) #prediction

perf_cart_train = performanceMeasures(prob_cart_train[,2], train_data$target, name="Decision Trees Training")
perf_cart_train
perf_cart_test = performanceMeasures(prob_cart_test[,2], val_data$target, name="Decision Trees Testing")
perf_cart_test

###NEURAL NETWORKS### ##DONE
set.seed(69)
nn = nnet(target~age+sex+cp+trestbps+chol+fbs+restecg+thalach
          +exang+oldpeak+slope+ca+thal, data = train_data, size = 5, decay = 5e-4, maxit = 100)

prob_nn_train = predict(nn, train_data)
prob_nn_test = predict(nn, val_data) #prediction

perf_nn_train = performanceMeasures(prob_nn_train, train_data$target, name="Neural Networks Training")
perf_nn_train
perf_nn_test = performanceMeasures(prob_nn_test, val_data$target, name="Neural Networks Testing")
perf_nn_test

###Summary Replication###
summary_rep_train=  rbind( perf_log_train, perf_cart_train, perf_nn_train)
summary_rep_train
summary_rep_test = rbind( perf_log_test, perf_cart_test, perf_nn_test)
summary_rep_test

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
rf = randomForest(as.factor(target)~age+sex+cp+trestbps+chol+fbs+restecg+thalach
                  +exang+oldpeak+slope+ca+thal, nodesize = 1 ,data = train_data, proximity = TRUE)

prob_rf_train = predict(rf, train_data, type='prob') # returns probabilities not 0,1 values with type="prob"
prob_rf_test = predict(rf, val_data, type='prob')

perf_rf_train = performanceMeasures(prob_rf_train[,2], train_data$target, name="Random Forest Training")
perf_rf_train
perf_rf_test = performanceMeasures(prob_rf_test[,2], val_data$target, name="Random Forest Testing")
perf_rf_test

### Summary Additional methods####

summary_add_train=  rbind(perf_cart_train, perf_bagtree_train, perf_rf_train)
summary_add_train
summary_add_test = rbind( perf_cart_test, perf_bagtree_test, perf_rf_test)
summary_add_test

######################
####Cross-validated###
######################

k = 100
cv_prob_log_train =  NULL
cv_prob_log_test = NULL
cv_perf_log_train= as.data.frame( matrix(0, ncol = 0, nrow = 0))
cv_perf_log_test=as.data.frame( matrix(0, ncol = 0, nrow = 0))
cv_prob_cart_train= NULL
cv_prob_cart_test= NULL
cv_perf_cart_train= as.data.frame( matrix(0, ncol = 0, nrow = 0))
cv_perf_cart_test=as.data.frame( matrix(0, ncol = 0, nrow = 0))
cv_prob_nn_train= NULL
cv_prob_nn_test= NULL
cv_perf_nn_train= as.data.frame( matrix(0, ncol = 0, nrow = 0))
cv_perf_nn_test=as.data.frame( matrix(0, ncol = 0, nrow = 0))
cv_prob_rf_train= NULL
cv_prob_rf_test= NULL
cv_perf_rf_train= as.data.frame( matrix(0, ncol = 0, nrow = 0))
cv_perf_rf_test=as.data.frame( matrix(0, ncol = 0, nrow = 0))
cv_prob_bagtree_train= NULL
cv_prob_bagtree_test= NULL
cv_perf_bagtree_train= as.data.frame( matrix(0, ncol = 0, nrow = 0))
cv_perf_bagtree_test=as.data.frame( matrix(0, ncol = 0, nrow = 0))

set.seed(345) # for replication of the cross validation
for (i in 1:k){ #sample randomly 100 times
  n = nrow(data)
  data_temp = data[sample(n),] 
  rand_rows = sample(1:nrow(data_temp), 0.7*nrow(data_temp)) #split into training and validation dataset
  train_data = data_temp[rand_rows, ]
  val_data = data_temp[-rand_rows, ]
  
  ##LOGISTISTIC ##DONE
  logit = glm(fmla, data = train_data, family = "binomial")
  try({
    cv_prob_log_train = predict(logit, train_data, type = "response")
    cv_prob_log_test = predict(logit, val_data, type = "response")
    cv_perf_log_train = rbind(cv_perf_log_train, performanceMeasures(cv_prob_log_train, train_data$target, name="Logit Training"))
    cv_perf_log_test = rbind(cv_perf_log_test,performanceMeasures(cv_prob_log_test, val_data$target, name="Logit Testing"))
  },TRUE)
  cart = rpart(fmla, data = train_data, method = 'class')
  try({
    cv_prob_cart_train = predict(cart, train_data)
    cv_prob_cart_test = predict(cart, val_data) #prediction
    cv_perf_cart_train = rbind(cv_perf_cart_train,performanceMeasures(cv_prob_cart_train[,2], train_data$target, name="DECISION TREES Training"))
    cv_perf_cart_test =rbind(cv_perf_cart_test, performanceMeasures(cv_prob_cart_test[,2], val_data$target, name="DECISION TREES Testing"))
    },TRUE)
  nn = nnet(target~age+sex+cp+trestbps+chol+fbs+restecg+thalach
            +exang+oldpeak+slope+ca+thal, data = train_data, size = 5, decay = 5e-4, maxit = 100)
  try({
    cv_prob_nn_train = predict(nn, train_data)
    cv_prob_nn_test = predict(nn, val_data) #prediction
    cv_perf_nn_train =rbind(cv_perf_nn_train, performanceMeasures(cv_prob_nn_train, train_data$target, name="NEURAL NETWORKS Training"))
    cv_perf_nn_test = rbind(cv_perf_nn_test, performanceMeasures(cv_prob_nn_test, val_data$target, name="NEURAL NETWORKS Testing"))
    },TRUE)
  rf = randomForest(as.factor(target)~age+sex+cp+trestbps+chol+fbs+restecg+thalach
                    +exang+oldpeak+slope+ca+thal, nodesize = 1 ,data = train_data, proximity = TRUE)
  try({
    cv_prob_rf_train = predict(rf, train_data, type='prob') # returns probabilities not 0,1 values with type="prob"
    cv_prob_rf_test = predict(rf, val_data, type='prob')
    cv_perf_rf_train = rbind(cv_perf_rf_train,performanceMeasures(cv_prob_rf_train[,2], train_data$target, name="RANDOM FOREST Training"))
    cv_perf_rf_test = rbind(cv_perf_rf_test,performanceMeasures(cv_prob_rf_test[,2], val_data$target, name="RANDOM FOREST Testing"))
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
    cv_prob_bagtree_train = predict.bag(treelist, newdata=train_data)
    cv_prob_bagtree_train= as.data.frame(cv_prob_bagtree_train)
    cv_prob_bagtree_train_1= rowSums(cv_prob_bagtree_train[, seq(2, ncol(cv_prob_bagtree_train), 2)])/length(treelist)
    
    cv_prob_bagtree_test = predict.bag(treelist, newdata=val_data)
    cv_prob_bagtree_test= as.data.frame(cv_prob_bagtree_test)
    cv_prob_bagtree_test_1= rowSums(cv_prob_bagtree_test[, seq(2, ncol(cv_prob_bagtree_test), 2)])/length(treelist)
    
    cv_perf_bagtree_train = rbind(cv_perf_bagtree_train, performanceMeasures(cv_prob_bagtree_train_1, train_data$target, name="Bagged Trees Training"))
    cv_perf_bagtree_test = rbind(cv_perf_bagtree_test, performanceMeasures(cv_prob_bagtree_test_1, val_data$target, name="Bagged Trees Testing"))
    },TRUE)
}

summary(cv_perf_log_train)
summary(cv_perf_log_test)
apply(cv_perf_log_test[,-1], 2, sd) # Standard deviation of performance measure values in test set

summary(cv_perf_cart_train)
summary(cv_perf_cart_test)
apply(cv_perf_cart_test[,-1], 2, sd)

summary(cv_perf_nn_train)
summary(cv_perf_nn_test)
apply(cv_perf_nn_test[,-1], 2, sd)

summary(cv_perf_bagtree_train)
summary(cv_perf_bagtree_test)
apply(cv_perf_bagtree_test[,-1], 2, sd)

summary(cv_perf_rf_train)
summary(cv_perf_rf_test)
apply(cv_perf_rf_test[,-1], 2, sd)

cv_summary_train =rbind(summary(cv_perf_log_train),summary(cv_perf_cart_train), summary(cv_perf_nn_train), summary(cv_perf_bagtree_train), summary(cv_perf_rf_train))
cv_summary_test =rbind(summary(cv_perf_log_test),summary(cv_perf_cart_test), summary(cv_perf_nn_test), summary(cv_perf_bagtree_test), summary(cv_perf_rf_test))

boxplot(cbind(cv_perf_log_test$Accuracy, cv_perf_cart_test$Accuracy, 
              cv_perf_nn_test$Accuracy, cv_perf_bagtree_test$Accuracy, cv_perf_rf_test$Accuracy), 
        names= c("Logit", "Decision Trees", "Neural Networks", "Bagged Trees", "Random Forest"),
        main= "Accuracy Cross Validation")

boxplot(cbind(cv_perf_log_test$error ,cv_perf_cart_test$error, 
              cv_perf_nn_test$error, cv_perf_bagtree_test$error,
              cv_perf_rf_test$error),
        names= c("Logit", "Decision Trees", "Neural Networks", "Bagged Trees", "Random Forest"),
        main= "Error Cross Validation")

boxplot(cbind(cv_perf_log_test$Area_under_the_curve ,cv_perf_cart_test$Area_under_the_curve, 
              cv_perf_nn_test$Area_under_the_curve, cv_perf_bagtree_test$Area_under_the_curve,
              cv_perf_rf_test$Area_under_the_curve),
        names= c("Logit", "Decision Trees", "Neural Networks", "Bagged Trees", "Random Forest"),
        main= "Area under the Curve Cross Validation")

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

prob_xg_train = predict(xgmodel,traintask) 
prob_xg_test = predict(xgmodel,testtask)

perf_xg_train = performanceMeasures(prob_xg_train$data$prob.1, prob_xg_train$data$truth, name="Gradient Boosting Training")
perf_xg_train
perf_xg_test  = performanceMeasures(prob_xg_test$data$prob.1, prob_xg_test$data$truth, name="Gradient Boosting Testing")
perf_xg_test

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
nnetmodel = train(learner = lrn_tune, task = traintask)
nnetpred = predict(nnetmodel,testtask)

#<<<<<<< HEAD
performanceMeasures(nnetpred$data$prob.1,nnetpred$data$truth)

auc = generateThreshVsPerfData(nnetpred, measures = list(fpr, tpr, mmce))
plotROCCurves(auc)
auc_nn = mlr::performance(nnetpred, mlr::auc)
#=======
prob_nnTu_train = predict(nnetmodel,traintask) 
prob_nnTu_test = predict(nnetmodel,testtask)

perf_nnTu_train = performanceMeasures(prob_nnTu_train$data$prob.1, prob_nnTu_train$data$truth, name="Neural Networks Tuned Training")
perf_nnTu_train
perf_nnTu_test  = performanceMeasures(prob_nnTu_test$data$prob.1, prob_nnTu_test$data$truth,name="Neural Networks Tuned Testing")
perf_nnTu_test

### Summary Parameter Tuning ###
summary_tun_train=  rbind(perf_xg_train, perf_nn_train, perf_nnTu_train)
summary_tun_train
summary_tun_test = rbind(perf_xg_test, perf_nn_test, perf_nnTu_test)
summary_tun_test
#>>>>>>> 3dc95572e27acf7b333de3a3784ed1d16ce7562a

################
###STATISTICS###
################
summary_rep_train
summary_rep_test

summary_add_train
summary_add_test

cv_summary_train 
cv_summary_test

summary_tun_train
summary_tun_test