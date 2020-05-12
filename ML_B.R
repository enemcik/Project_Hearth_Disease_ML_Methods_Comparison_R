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

library(party)
library(openxlsx)
library(DiagrammeR)
library(GGally)
library(neuralnet)
library(openxlsx)
library(foreign)
library(readstata13)
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

####DATA HANDLING - NAs/Zeros
data = read.xlsx("ML_TRY.xlsx")
data1 = as.data.frame(data)
is.data.frame(data1)
head(data1,5)

colnames(data1)
data1$Region.ADL = as.factor(data1$Region.ADL)
data1$Tarif = as.factor(data1$Tarif)
data1$Instalatie = as.character(data1$Instalatie)
data1$Result = as.factor(data1$Result)
data1$Cod.CAEN = as.factor(data1$Cod.CAEN)
#data1$Suspected.Tempering = as.factor(data1$Suspected.Tempering)
#names(data1)[names(data1) == "AVG/GROUP"] = "GROUP"

apply(data1, 2, function(x) {sum(is.na(x))})
#data1 = data1[ , -which(names(data1) %in% c("M1","M2", "M3"))]
#data_NA = data1 %>% drop_na()

####DATA TRANSFORMATION###

vars = c("Result","Tarif","S1","S2","S3","S4", "S5", "DEV1","DEV2","DEV3","DEV4","DEV5","MAX","AVG", "Cod.CAEN", "CodEchip")
data_ready = data1[,vars]
head(data_ready)
summary(data_ready)

n_09 = nrow(data_ready) #shuffling the data
data_ready = data_ready[sample(n_09),] 

row.names(data_ready) = data_ready[,"CodEchip"]
data_ready = data_ready[,-16]

###TRAIN/TEST DATASET
vars_n = vars[c(-1,-2,-15,-16)]
rand_rows = sample(1:nrow(data_ready), 0.8*nrow(data_ready))
train_data = data_ready[rand_rows, ]
test_data = data_ready[-rand_rows, ]

y = "Result" #creating a formula
fmla = paste(y, paste(vars_n, collapse="+"),sep="~")
fmla

########################
### GRADIENT BOOSTING ###

label = train_data$Result
label = as.vector(label)
str(label)

mat = train_data[,c(-1,-15)]
head(mat)
mat = data.table(mat)
sparseMat = sparsify(mat)
rownames(sparseMat) = rownames(train_data)
head(sparseMat)

mat_test = test_data[,c(-1,-15)]
head(mat_test)
mat_test = data.table(mat_test)
mat_test = sparsify(mat_test)
rownames(mat_test) = rownames(test_data)
head(mat_test)

mat_train = train_data[,c(-1,-15)]
mat_train = data.table(mat_train)
mat_train = sparsify(mat_train)
rownames(mat_train) = rownames(train_data)
head(mat_train)

#bstSparse = xgboost(data = sparseMat, label = label, max.depth = 10, eta = 1, nthread = 2, nrounds = 10, objective = "binary:logistic",verbose = 2)

#dtrain = xgb.DMatrix(data = sparseMat, label = label)
#bstDMatrix = xgboost(data = dtrain, max.depth = 10, eta = 1, nthread = 2, nrounds = 10, objective = "binary:logistic")

bst_try = xgboost(data = as.matrix(sparseMat), label = label, max.depth = 15, eta = 2, nthread = 2, nrounds = 5, objective = "binary:logistic")
xgb.plot.tree(model = bst_try)

bst = xgboost(data = as.matrix(sparseMat), label = label, max.depth = 15, eta = 2, nthread = 2, nrounds = 100, objective = "binary:logistic")
xgb.plot.tree(model = bst)

pred = predict(bst, mat_test)
print(length(pred))
prediction = as.numeric(pred > 0.1)
print(head(prediction,30))
test_data$prediction = prediction

confusionMatrix(table(test_data$Result,test_data$prediction), positive = "1")

auc(test_data$Result, as.numeric(test_data$prediction))

plot(roc(test_data$Result, as.numeric(test_data$prediction), direction="<"),
     col="red", lwd=3, main="ROC")


########################
### XGBOOST ALL DATA ###

label_all = data_ready$Result
label_all = as.vector(label_all)
str(label_all)

mat_all = data_ready[,vars_n]
head(mat_all)
mat_all = data.table(mat_all)
sparseMat_all = sparsify(mat_all)
rownames(sparseMat_all) = rownames(data_ready)
head(sparseMat_all)

#bstSparse = xgboost(data = sparseMat, label = label, max.depth = 10, eta = 1, nthread = 2, nrounds = 10, objective = "binary:logistic",verbose = 2)

#dtrain = xgb.DMatrix(data = sparseMat, label = label)
#bstDMatrix = xgboost(data = dtrain, max.depth = 10, eta = 1, nthread = 2, nrounds = 10, objective = "binary:logistic")

bst_all = xgboost(data = as.matrix(sparseMat_all), label = label_all, max.depth = 7, eta = 2, nthread = 2, nrounds = 10, objective = "binary:logistic", missing = NA)
xgb.plot.tree(model = bst_all)

pred_all = predict(bst_all, sparsify(mat_all))
print(length(pred_all))
prediction_all = as.numeric(pred_all > 0.5)
print(head(prediction_all,30))
data_ready$prediction = prediction_all

confusionMatrix(table(data_ready$Result,data_ready$prediction), positive = "1")

auc(data_ready$Result, as.numeric(data_ready$prediction))

pred_round = round(pred_all,digits=4)
pred_round = as.data.frame(pred_round)
pred_round$Codechip = rownames(data_ready)
write.xlsx(pred_round, "results_xgboost.xlsx")


#########
### RF ###
vars_rf = c("Result","S1","S2","S3","S4", "S5", "DEV1","DEV2","DEV3","DEV4","DEV5","MAX","AVG","Cod.CAEN")
vars_rf_r = c("S1","S2","S3","S4", "S5", "DEV1","DEV2","DEV3","DEV4","DEV5","MAX","AVG","Cod.CAEN")

all_data_rf = na.approx(data_ready[,vars_rf])
rownames(all_data_rf) = rownames(data_ready)
all_data_rf = na.omit(all_data_rf)
nrow(all_data_rf)
all_data_rf = as.data.frame(all_data_rf)

rf_all = randomForest(x=all_data_rf[,vars_rf_r],y=all_data_rf$Result,ntree = 33, nodesize =7, importance = T)
pred_all = predict(rf_all,newdata = all_data_rf[vars_rf_r], type = "response")
prediction_all = as.numeric(pred_all > 0.5)
all_data_rf$prediction = prediction_all
all_data_rf$prediction_prob = pred_all


confusionMatrix(table(all_data_rf$Result,all_data_rf$prediction), positive = "1")

auc(all_data_rf$Result, as.numeric(all_data_rf$prediction))

pred_round = round(pred_all,digits=4)
pred_round$Codechip = rownames(pred_round)
pred_round = as.data.frame(pred_round)
write.xlsx(pred_round, "results_rf.xlsx")


########################
### NEURAL NETWORKS ###

neural_fmla = paste(y, paste(vars_n, collapse="+"),sep="~")

scale01 <- function(x){  #hard with NAs
  (x - min(x)) / (max(x) - min(x))
}

neural_data = data_ready[c(-2,-15)]
#neural_data$Result = as.numeric(neural_data$Result)
#maxmindf = as.data.frame(lapply(neural_data,scale01)) hard with NAs
scaled_neural = as.data.frame(scale(neural_data[-1]))
scaled_neural$Result = neural_data$Result
scaled_neural_n = na.approx(scaled_neural)
scaled_neural_n = as.data.frame(scaled_neural_n)
row.names(scaled_neural_n) = row.names(scaled_neural)
head(scaled_neural_n)

train_neural = scaled_neural_n[rand_rows, ]
test_neural = scaled_neural_n[-rand_rows, ]

nn = neuralnet(neural_fmla, data = train_neural, hidden=c(10,5), linear.output=FALSE, threshold=0.1, algorithm = "rprop+")
plot(nn)

temp_test = subset(test_neural, select = vars_n)
head(temp_test)

nn.results = neuralnet::compute(nn,temp_test)
results = data.frame(actual = test_neural$Result, prediction = nn.results$net.result)
results

prediction_try = nn.results$net.result

conversion = function(x){
  if(x>0.08){
    x = 1
  } else { x = 0
  }
}

tryresults = sapply(prediction_try,conversion)
confusionMatrix(table(results$actual,tryresults), positive = "1")

auc(actual, prediction)

plot(roc(actual, prediction, direction="<"),
     col="red", lwd=3, main="ROC")

##########################
### NNs with all data ###

nn_all = neuralnet(neural_fmla, data = scaled_neural_n, hidden=c(7,2), linear.output=FALSE, threshold=0.1, algorithm = "rprop+")
plot(nn_all)

temp_test_all = subset(scaled_neural_n, select = vars_n)
head(temp_test_all)

nn_all.results = neuralnet::compute(nn_all,temp_test_all)
results_all = data.frame(actual = scaled_neural_n$Result, prediction = nn_all.results$net.result)
results_all

prediction_try_all = nn_all.results$net.result

conversion_all = function(x){
  if(x>0.08){
    x = 1
  } else { x = 0
  }
}

tryresults_all = sapply(prediction_try_all,conversion_all)
confusionMatrix(table(results_all$actual,tryresults_all), positive = "1")

auc(actual, prediction)

##SENSITIVITY PLAY
prediction_try = nn.results$net.result

conversion = function(x){
  if(x>0.25){
    x = 1
  } else { x = 0
  }
}

tryresults = sapply(prediction_try,conversion)
confusionMatrix(table(actual,tryresults), positive = "1")

table(actual,tryresults)

auc(actual, tryresults)

plot(roc(actual, tryresults, direction="<"),
     col="red", lwd=3, main="ROC")

err_try = mean(tryresults != actual)
print(paste("test-error=", err_try))


######### ######### ######### ######### ######### ######### ######### ######### ######### ######### #########
######## SMOTEd dataset #########

balanced_data = SMOTE(Result ~., train_data, perc.over = 300, perc.under = 150)
table(balanced_data$Result)

label_b = balanced_data$Result
label_b = as.vector(label_b)
str(label_b)

mat_b = balanced_data[,-1]
head(mat_b)
mat_b = data.table(mat_b)
sparseMat_b = sparsify(mat_b)
rownames(sparseMat_b) = rownames(balanced_data)
head(sparseMat_b)

bst_b = xgboost(data = as.matrix(sparseMat_b), label = label_b, max.depth = 5, eta = 2, nthread = 2, nrounds = 10, objective = "binary:logistic")

colnames(mat_test)
colnames(sparseMat_b)

pred_b = predict(bst_b, mat_test)
print(length(pred_b))
prediction_b = as.numeric(pred_b > 0.5)
print(head(prediction_b,30))
test_data$prediction_b = prediction_b

table(test_data$Result,test_data$prediction_b)

err = mean(as.numeric(pred_b > 0.5) != test_data$Result)
print(paste("test-error=", err))

auc(test_data$Result, as.numeric(test_data$prediction_b))

plot(roc(test_data$Result, as.numeric(test_data$prediction_b), direction="<"),
     col="red", lwd=3, main="ROC")

##Prediction on TRAIN
pred_train = predict(bst, mat_train)
print(length(pred_train))
prediction_train = as.numeric(pred_train > 0.5)
print(head(prediction_train,30))
train_data$prediction = prediction_train

table(train_data$Result,train_data$prediction)

err_train = mean(as.numeric(pred_train > 0.5) != train_data$Result)
print(paste("test-error=", err_train))

auc(train_data$Result, as.numeric(train_data$prediction))

plot(roc(train_data$Result, as.numeric(train_data$prediction), direction="<"),
     col="red", lwd=3, main="ROC")


##################
### PREDICTION ###

####DATA HANDLING - NAs/Zeros
data_vranc = read.xlsx("ML_NEW.xlsx")
data2 = as.data.frame(data_vranc)
is.data.frame(data2)
head(data2,5)

colnames(data2)
data2$Region.ADL = as.factor(data2$Region.ADL)
data2$Tarif = as.factor(data2$Tarif)
data2$Instalatie = as.character(data2$Instalatie)
data2$Result = as.factor(data2$Result)
data2$Cod.CAEN = as.factor(data2$Cod.CAEN)
#data1$Suspected.Tempering = as.factor(data1$Suspected.Tempering)
#names(data1)[names(data1) == "AVG/GROUP"] = "GROUP"

apply(data2, 2, function(x) {sum(is.na(x))})
#data1 = data1[ , -which(names(data1) %in% c("M1","M2", "M3"))]
#data_NA = data1 %>% drop_na()

data_ready_vranc = data2[,vars]
head(data_ready_vranc)
rownames(data_ready_vranc) = data_ready_vranc$CodEchip
summary(data_ready_vranc)

### XGBOOST

mat_vranc = data_ready_vranc[,vars_n]
head(mat_vranc)
mat_vranc = data.table(mat_vranc)
sparseMat_vranc = sparsify(mat_vranc)
rownames(sparseMat_vranc) = rownames(data_ready_vranc)
head(sparseMat_vranc)

pred_vranc = predict(bst_all, sparsify(mat_vranc))

pred_final = round(pred_vranc,digits=4)
pred_final = as.data.frame(pred_final)
pred_final$Codechip = rownames(data_ready_vranc)

write.xlsx(pred_final, "results_xgboost_vranc.xlsx")

### RANDOM FORREST

all_data_vranc = na.approx(data_ready_vranc[,vars_rf_r])
rownames(all_data_vranc) = rownames(data_ready_vranc)
all_data_vranc = na.omit(all_data_vranc)
nrow(all_data_vranc)
all_data_vranc = as.data.frame(all_data_vranc)

pred_vrancea = predict(rf_all,newdata = all_data_vranc[vars_rf_r], type = "response")

pred_fin = round(pred_vrancea,digits=4)
pred_fin = as.data.frame(pred_fin)
pred_fin$CodEchip = rownames(pred_fin)

write.xlsx(pred_fin, "results_rf_vranc.xlsx")
