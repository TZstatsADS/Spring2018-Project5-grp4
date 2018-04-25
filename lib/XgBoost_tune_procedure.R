setwd("C:/Users/Kevin Zhang/Documents/GitHub/Spring2018-Project5-grp_4/data")
# load packages
library("xgboost")
library("haven")
library("tidyverse") 
library("mlr")
library("plyr")
library("ggplot2")
library("ModelMetrics")
library("caret")
library("pROC")
library("ROCR")
######################
# Data Preparation
######################

# Hour Column Data Processing Function
TimeToNum <- function(x)
{
  Hour <- as.numeric(substr(x, start = 12, stop = 13)) * 2
  HourHalf <- as.numeric(substr(x, start = 15, stop = 15))
  if (HourHalf >= 3)
    Hour <- Hour + 1
  return(Hour)
}
timestart<-Sys.time()

# read data for processed train data
processed.data <- read.csv("Train0422.csv")
processed.data <- apply(processed.data[,-c(7,8)], 2, as.numeric)
train.data  <- as.matrix(processed.data[,c("ip","app","device","os","channel","Hour")])
train.class <- as.matrix(processed.data[,"is_attributed"])

# separate train data
train.data1 <- train.data[(1:nrow(train.class))[train.class != 1],]
train.data2 <- train.data[(1:nrow(train.class))[train.class == 1],]
train.class1 <- train.class[(1:nrow(train.class))[train.class == 0]]
train.class2 <- train.class[(1:nrow(train.class))[train.class != 0]]

# load test data
raw.data1 <- read.csv("test.csv")
raw.data1$Hour <- mapply(TimeToNum, raw.data1$click_time)
colnames(raw.data1)
raw.data1 <- apply(raw.data1[,-1], 2, as.numeric)
test.data  <- as.matrix(raw.data1[,c("ip","app","channel","Hour")])

###################################
###################################
# Method 1 XgBoost Cross Validation
###################################
###################################

###################################
# Step 1 CV for the nround in Xgboost 
###################################

# data preparation for xgboost
xgb.train.data <- xgb.DMatrix(data = train.data,label = train.class)
xgb.test.data <- xgb.DMatrix(data = test.data)

# Default Parameter 
xgb_params <- list("objective" = "binary:logistic",  # logistic regression for binary classification
                   "eval_metric" = "auc",          
                   "silent"="0")

# CV of xgboost (selection of nrounds)
nround    <- 1000 # number of XGBoost rounds; most of the situations, nrounds is less than 100. Therefore, 200 should be enough for the cross validation.
cv.nfold  <- 5

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params      = xgb_params,
                   data        = xgb.train.data, 
                   nrounds     = nround,
                   nfold       = cv.nfold,
                   verbose     = TRUE,
                   prediction  = TRUE,
                   tree_method = 'exact')

max_auc = max(cv_model[["evaluation_log"]][, 4])     
max_auc_index = max((1:nround)[cv_model[["evaluation_log"]][, 4] == max_auc])   # optimal nrounds in the train model
max_auc_index <- 100

# train model on all the train dataset based on the optimal nrounds
model1 <- xgb.train(data = xgb.train.data,    
                    nround = max_auc_index, 
                    params = xgb_params,
                    tree_method = 'exact')

##############################################
# Test prediction 
test.data.pred <- predict(model1,newdata = xgb.test.data)
output.matrix <- data.frame(test.data.pred)
output.matrix <- cbind(raw.data1[,"click_id"],output.matrix)
colnames(output.matrix) <- c("click_id", "is_attributed")
#write.csv(output.matrix,file = "pred.csv",row.names = FALSE)

timeend1<-Sys.time()
runningtime1<-timeend1-timestart
print(runningtime1) 


#########################
# Step 2: cv for max_depth
#########################

max.depth <- c(seq(from = 1, to = 6, by = 1))        
nd <- length(max.depth)
# Define SVM RBF Kernel cross validation function
k = 5
# # Random index of the train data
# index1 <- sample (1:nrow(train.data1))
# index2 <- sample (1:nrow(train.data2))
# # The first and last index of each folds
# first1 <- round(seq(0,(k-1)) * length(index1)/k)+1
# last1 <- round (seq(1,k) * length(index1)/k)
# first2 <- round(seq(0,(k-1)) * length(index2)/k)+1
# last2 <- round (seq(1,k) * length(index2)/k)
# For the combination of cost and gamma, I store it in the matrix.
# Random index of the train data
index <- sample(1:nrow(train.data))
# The first and last index of each folds
first <- round(seq(0,(k-1)) * length(index)/k)+1
last <- round (seq(1,k) * length(index)/k)


cv.max.depth.err.vec <- c(rep(0,nd))
for(i in 1:nd){
  print(i)
  for(j in 1:k){
    print(j)
    validion_index <- index[(first[j]:last[j])]
    
    xgb.cv.train.data <- xgb.DMatrix(data = train.data[-validion_index,],
                                     label = train.class[-validion_index,])
    xgb.cv.test.data <- xgb.DMatrix(data = train.data[validion_index,])
    
    cv.max.depth.model <- xgb.train(data = xgb.cv.train.data,    
                                    nround = max_auc_index, 
                                    params = xgb_params,
                                    max_depth = max.depth[i],
                                    tree_method = 'exact')
    train.data.pred <- predict(cv.max.depth.model,newdata = xgb.cv.test.data)
    pred <- prediction(train.data.pred,train.class[validion_index,])
    unique(train.class[validion_index,])
    roc.perf = performance(pred, measure = "auc")
    roc.perf@y.values[[1]]
    cv.max.depth.err.vec[i] <- cv.max.depth.err.vec[i] + roc.perf@y.values[[1]]
  }
}
cv.max.depth.err.vec <- cv.max.depth.err.vec/k
max.depth.optimal <- max.depth[which.max(cv.max.depth.err.vec)]

# train model on all the train dataset based on the optimal nrounds
model1 <- xgb.train(data = xgb.train.data,    
                    nround = max_auc_index, 
                    params = xgb_params,
                    tree_method = 'exact',
                    max_depth = max.depth.optimal)


# Test Result 
test.data.pred <- predict(model1,newdata = xgb.test.data)
output.matrix <- data.frame(test.data.pred)
output.matrix <- cbind(raw.data1[,"click_id"],output.matrix)
colnames(output.matrix) <- c("click_id", "is_attributed")
write.csv(output.matrix,file = "pred.csv",row.names = FALSE)



#########################
# Step 3: cv for subsample
#########################

subsample <- c(seq(from = 0.5, to = 1, by = 0.1))        
nd <- length(subsample)
# Define XgB cross validation function
k = 5
# Random index of the train data
index <- sample(1:nrow(train.data))
# The first and last index of each folds
first <- round(seq(0,(k-1)) * length(index)/k)+1
last <- round (seq(1,k) * length(index)/k)
# For the combination of cost and gamma, I store it in the matrix.
cv.subsample.err.vec <- c(rep(0,nd))
for(i in 1:nd){
  print(i)
  for(j in 1:k){
    print(j)
    validion_index <- index[(first[j]:last[j])]
    
    xgb.cv.train.data <- xgb.DMatrix(data = train.data[-validion_index,],
                                     label = train.class[-validion_index,])
    xgb.cv.test.data <- xgb.DMatrix(data = train.data[validion_index,])
    
    cv.subsample.model <- xgb.train(data = xgb.cv.train.data,    
                                    nround = max_auc_index, 
                                    params = xgb_params,
                                    max_depth = max.depth.optimal,
                                    subsample = subsample[i],
                                    tree_method = 'exact')
    train.data.pred <- predict(cv.subsample.model,newdata = xgb.cv.test.data)
    pred <- prediction(train.data.pred,train.class[validion_index,])
    roc.perf = performance(pred, measure = "auc")
    cv.subsample.err.vec[i] <- cv.subsample.err.vec[i] + roc.perf@y.values[[1]]
  }
}
cv.subsample.err.vec <- cv.subsample.err.vec/k
subsample.optimal <- subsample[which.max(cv.subsample.err.vec)]

model2 <- xgb.train(data = xgb.train.data,    
                    nround = max_auc_index, 
                    params = xgb_params,
                    tree_method = 'exact',
                    max_depth = max.depth.optimal,
                    subsample = subsample.optimal)
test.data.pred <- predict(model2,newdata = xgb.test.data)
output.matrix <- data.frame(test.data.pred)
output.matrix <- cbind(raw.data1[,"click_id"],output.matrix)
colnames(output.matrix) <- c("click_id", "is_attributed")
write.csv(output.matrix,file = "pred_subsample.csv",row.names = FALSE)



#########################
# Step 4: cv for min_child_weight
#########################

min_child_weight <- c(seq(from = 0, to = 9, by = 1))        
nd <- length(min_child_weight)
# Define SVM RBF Kernel cross validation function
k = 5
# Random index of the train data
index <- sample(1:nrow(train.data))
# The first and last index of each folds
first <- round(seq(0,(k-1)) * length(index)/k)+1
last <- round (seq(1,k) * length(index)/k)
# For the combination of cost and gamma, I store it in the matrix.
cv.min.child.weight.err.vec <- c(rep(0,nd))
for(i in 1:nd){
  print(i)
  for(j in 1:k){
    print(j)
    validion_index <- index[(first[j]:last[j])]
    
    xgb.cv.train.data <- xgb.DMatrix(data = train.data[-validion_index,],
                                     label = train.class[-validion_index,])
    xgb.cv.test.data <- xgb.DMatrix(data = train.data[validion_index,])
    
    cv.min.child.weight.model <- xgb.train(data = xgb.cv.train.data,    
                                           nround = max_auc_index, 
                                           params = xgb_params,
                                           max_depth = max.depth.optimal,
                                           subsample = subsample.optimal,
                                           min_child_weight = min_child_weight[i],
                                           tree_method = 'exact')
    train.data.pred <- predict(cv.min.child.weight.model,newdata = xgb.cv.test.data)
    pred <- prediction(train.data.pred,train.class[validion_index,])
    roc.perf = performance(pred, measure = "auc")
    cv.min.child.weight.err.vec[i] <- cv.min.child.weight.err.vec[i] + roc.perf@y.values[[1]]
  }
}
cv.min.child.weight.err.vec <- cv.min.child.weight.err.vec/k
min_child_weight.optimal <- min_child_weight[which.max(cv.min.child.weight.err.vec)]


model3 <- xgb.train(data = xgb.train.data,    
                    nround = max_auc_index, 
                    params = xgb_params,
                    tree_method = 'exact',
                    max_depth = max.depth.optimal,
                    subsample = subsample.optimal,
                    min_child_weight = min_child_weight.optimal)
test.data.pred <- predict(model3,newdata = xgb.test.data)
output.matrix <- data.frame(test.data.pred)
output.matrix <- cbind(raw.data1[,"click_id"],output.matrix)
colnames(output.matrix) <- c("click_id", "is_attributed")
write.csv(output.matrix,file = "pred_min_child_weight.csv",row.names = FALSE)


#########################
# Step 5: cv for colsample_bytree
#########################

colsample_bytree <- c(seq(from = 0.5, to = 1, by = 0.1))        
nd <- length(colsample_bytree)
# Define SVM RBF Kernel cross validation function
k = 5
# Random index of the train data
index <- sample(1:nrow(train.data))
# The first and last index of each folds
first <- round(seq(0,(k-1)) * length(index)/k)+1
last <- round (seq(1,k) * length(index)/k)
# For the combination of cost and gamma, I store it in the matrix.
cv.colsample.bytree.err.vec <- c(rep(0,nd))
for(i in 1:nd){
  print(i)
  for(j in 1:k){
    print(j)
    validion_index <- index[(first[j]:last[j])]
    
    xgb.cv.train.data <- xgb.DMatrix(data = train.data[-validion_index,],
                                     label = train.class[-validion_index,])
    xgb.cv.test.data <- xgb.DMatrix(data = train.data[validion_index,])
    
    cv.colsample.bytree.model <- xgb.train(data = xgb.cv.train.data,    
                                           nround = max_auc_index, 
                                           params = xgb_params,
                                           max_depth = max.depth.optimal,
                                           subsample = subsample.optimal,
                                           min_child_weight = min_child_weight.optimal,
                                           colsample_bytree = colsample_bytree[i],
                                           tree_method = 'exact')
    train.data.pred <- predict(cv.colsample.bytree.model,newdata = xgb.cv.test.data)
    pred <- prediction(train.data.pred,train.class[validion_index,])
    roc.perf = performance(pred, measure = "auc")
    cv.colsample.bytree.err.vec[i] <- cv.colsample.bytree.err.vec[i] + roc.perf@y.values[[1]]
  }
}

cv.colsample.bytree.err.vec <- cv.colsample.bytree.err.vec/k
colsample_bytree.optimal <- colsample_bytree[which.max(cv.colsample.bytree.err.vec)]

model4 <- xgb.train(data = xgb.train.data,    
                    nround = max_auc_index, 
                    params = xgb_params,
                    tree_method = 'exact',
                    max_depth = max.depth.optimal,
                    subsample = subsample.optimal,
                    min_child_weight = min_child_weight.optimal,
                    colsample_bytree = colsample_bytree.optimal)
test.data.pred <- predict(model4,newdata = xgb.test.data)
output.matrix <- data.frame(test.data.pred)
output.matrix <- cbind(raw.data1[,"click_id"],output.matrix)
colnames(output.matrix) <- c("click_id", "is_attributed")
write.csv(output.matrix,file = "pred2.csv",row.names = FALSE)




#########################
# Step 6: cv for eta
#########################

eta <- c(0.01,0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5)        
nd <- length(eta)
# Define SVM RBF Kernel cross validation function
k = 5
# Random index of the train data
index <- sample(1:nrow(train.data))
# The first and last index of each folds
first <- round(seq(0,(k-1)) * length(index)/k)+1
last <- round (seq(1,k) * length(index)/k)
# For the combination of cost and gamma, I store it in the matrix.
cv.eta.err.vec <- c(rep(0,nd))
for(i in 1:nd){
  print(i)
  for(j in 1:k){
    print(j)
    validion_index <- index[(first[j]:last[j])]
    
    xgb.cv.train.data <- xgb.DMatrix(data = train.data[-validion_index,],
                                     label = train.class[-validion_index,])
    xgb.cv.test.data <- xgb.DMatrix(data = train.data[validion_index,])
    
    cv.eta.model <- xgb.train(data = xgb.cv.train.data,    
                              nround = max_auc_index, 
                              params = xgb_params,
                              max_depth = max.depth.optimal,
                              subsample = subsample.optimal,
                              min_child_weight = min_child_weight.optimal,
                              colsample_bytree = colsample_bytree.optimal,
                              eta = eta[i],
                              tree_method = 'exact')
    train.data.pred <- predict(cv.eta.model,newdata = xgb.cv.test.data)
    pred <- prediction(train.data.pred,train.class[validion_index,])
    roc.perf = performance(pred, measure = "auc")
    cv.eta.err.vec[i] <- cv.eta.err.vec[i] + roc.perf@y.values[[1]]
  }
}

cv.eta.err.vec <- cv.eta.err.vec/k
eta.optimal <- eta[which.max(cv.eta.err.vec)]

model5 <- xgb.train(data = xgb.train.data,    
                    nround = max_auc_index, 
                    params = xgb_params,
                    tree_method = 'exact',
                    max_depth = max.depth.optimal,
                    subsample = subsample.optimal,
                    min_child_weight = min_child_weight.optimal,
                    colsample_bytree = colsample_bytree.optimal,
                    eta = eta.optimal)
test.data.pred <- predict(model5,newdata = xgb.test.data)
output.matrix <- data.frame(test.data.pred)
output.matrix <- cbind(raw.data1[,"click_id"],output.matrix)
colnames(output.matrix) <- c("click_id", "is_attributed")
write.csv(output.matrix,file = "pred3.csv",row.names = FALSE)



######################
# Step 7: rechoose the nrounds
######################
# Fit cv.nfold * cv.nround XGB models and save OOF predictions
nround <- 2000
cv_model <- xgb.cv(params = xgb_params,
                   data = xgb.train.data, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = TRUE,
                   prediction = TRUE,
                   max_depth = max.depth.optimal,
                   subsample = subsample.optimal,
                   min_child_weight = min_child_weight.optimal,
                   colsample_bytree = colsample_bytree.optimal,
                   eta = eta.optimal,
                   tree_method = 'exact')
min_logloss = max(cv_model[["evaluation_log"]][, 4])     # minimal log loss value
min_logloss_index = min((1:nround)[cv_model[["evaluation_log"]][, 4]==min_logloss])

# train model on all the train dataset based on the optimal nrounds
model <- xgb.train(data = xgb.train.data,    
                   nround = min_logloss_index, 
                   params = xgb_params,
                   max_depth = max.depth.optimal,
                   subsample = subsample.optimal,
                   min_child_weight = min_child_weight.optimal,
                   colsample_bytree = colsample_bytree.optimal,
                   eta = eta.optimal)

test.data.pred <- predict(model,newdata = xgb.test.data)
output.matrix <- data.frame(test.data.pred)
output.matrix <- cbind(raw.data1[,"click_id"],output.matrix)
colnames(output.matrix) <- c("click_id", "is_attributed")
write.csv(output.matrix,file = "pred4.csv",row.names = FALSE)

# The highest accuracy in each cross-validation
max.tuning.proc <- c(max(cv.max.depth.err.vec),max(cv.subsample.err.vec),max(cv.min.child.weight.err.vec)
                     ,max(cv.colsample.bytree.err.vec),max(cv.eta.err.vec))


timeend.mtd1 <-Sys.time()
runningtime2<-timeend.mtd1-timestart
print(runningtime2) 




#######################################################
#######################################################
# Method 2 Grid search procedure (Time-consuming)
#######################################################
#######################################################

# Selection of eta
# It controls the learning rate, i.e., the rate at which our model learns patterns in data.After every round, it shrinks the feature weights to reach the best optimum.
#eta.vec <- c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7)
#test.acc <- c()
#for (i in 1:7){
#  eta <- eta.vec[i]
#  xgb_params <- list("objective" = "multi:softmax",
#                     "eval_metric" = "mlogloss",
#                     "num_class" = numberOfClasses,
#                     "eta"=eta)
#  model <- xgb.train(data = xgb.train.data,    
#                     nround = min_logloss_index, 
#                     params = xgb_params)
#  test.data.pred <- predict(model,newdata = xgb.test.data)
#  test.acc[i] <- sum(test.data.pred==test.class[,2])/nrow(test.class)
#}
#test.acc.max = max(test.acc)
#eta_optimal = eta.vec[(1:8)[test.acc==test.acc.max]]

# 1. Create learner
learner <- makeLearner("classif.xgboost",predict.type = "prob")

#eta_optimal <- as.numeric(eta_optimal[1])
min_logloss_index <- as.numeric(min_logloss_index)
learner$par.vals <- list("booster" = "gbtree", "objective" = "multi:softprob",   #multi:softprob  multi:softmax
                         "eval_metric" = "mlogloss", "num_class" = "numberOfClasses",
                         "nrounds" = min_logloss_index) #
# 2. Set parameter space
params.space <- makeParamSet( makeIntegerParam("max_depth",lower = 3L,upper = 8L),   # makeDiscreteParam("booster",values = c("gbtree","gblinear"))
                              makeNumericParam("min_child_weight",lower = 3L,upper = 6L),
                              makeNumericParam("subsample",lower = 0.7,upper = 1), 
                              makeNumericParam("colsample_bytree",lower = 0.4,upper = 0.9),
                              makeNumericParam("eta",lower = 0.2,upper = 0.5))

# 3. Set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

# 4. search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)

# 5. set parallel backend
library("parallel")
library("parallelMap") 
parallelStartSocket(cpus = detectCores())

# 6. create tasks
train.data.feature.num <- apply(train.data.feature,2,as.numeric)
colnames(train.class) <- c("id","label")
train.data.feature.comb <- cbind(train.data.feature.num[,2:ncol(train.data.feature.num)],train.class[,2])
colnames(train.data.feature.comb)[1] <- "X1"
colnames(train.data.feature.comb)[ncol(train.data.feature.comb)] <- "label"
train.data.feature.comb <- as.data.frame(train.data.feature.comb)
train.data.feature.comb <- lapply(train.data.feature.comb, function(x) {
  if(is.factor(x)) as.numeric(as.character(x)) else x
})
train.data.feature.comb$label <- as.factor(train.data.feature.comb$label)
train.data.feature.comb <- as.data.frame(train.data.feature.comb)
traintask <- makeClassifTask (data = train.data.feature.comb, target = "label")

test.data.feature.num <- apply(test.data.feature,2,as.numeric)
colnames(test.class) <- c("id","label")
test.data.feature.comb <- cbind(test.data.feature.num[,2:ncol(test.data.feature.num)],test.class[,2])
colnames(test.data.feature.comb)[1] <- "X1"
colnames(test.data.feature.comb)[ncol(test.data.feature.comb)] <- "label"
test.data.feature.comb <- as.data.frame(test.data.feature.comb)
test.data.feature.comb <- lapply(test.data.feature.comb, function(x) {
  if(is.factor(x)) as.numeric(as.character(x)) else x
})
test.data.feature.comb$label <- as.factor(test.data.feature.comb$label)
test.data.feature.comb <- as.data.frame(test.data.feature.comb)
testtask <- makeClassifTask (data = test.data.feature.comb, target = "label")


# 6. parameter tuning
mytune <- tuneParams(learner = learner, task = traintask, resampling = rdesc, measures = mmce, 
                     par.set = params.space, control = ctrl, show.info = TRUE)
mytune$y 

# set hyperparameters
lrn_tune <- setHyperPars(learner,par.vals = mytune$x)

# train model
xgmodel <- train(learner = lrn_tune,task = traintask)

# predict model
xgpred <- predict(xgmodel,testtask)

# test error
sum(xgpred$data$response==test.class[,2])/nrow(test.class)





train.data[,1:4]



#################################
#################################
# Appendix 1: variable importance plot
#################################
#################################
mat <- xgb.importance (feature_names = colnames(train.data), model = model1)
xgb.plot.importance(importance_matrix = mat,rel_to_first = TRUE,col = "blue") 

#################################
#################################
# Appendix 2: train function and test function
#################################
#################################

# train function
train_xgboost <- function(traindata){
  # traindata has to be a matrix
  timestart <- Sys.time()
  # Data Preparation
  xgb.train.data <- xgb.DMatrix(data = traindata[,-1],label = traindata[,1] - 1)
  # Default Parameter 
  numberOfClasses <- length(unique(traindata[,1]))
  xgb_params <- list("objective" = "multi:softmax",
                     "eval_metric" = "mlogloss",
                     "num_class" = numberOfClasses,
                     "silent"="0")
  cv_model <- xgb.cv(params = xgb_params,
                     data = xgb.train.data, 
                     nrounds = 200,
                     nfold = 10,
                     verbose = FALSE,
                     prediction = TRUE)
  min_logloss = min(cv_model[["evaluation_log"]][, 4])     
  min_logloss_index = (1:200)[cv_model[["evaluation_log"]][, 4]==min_logloss]   
  xgb_fit <- xgb.train(data = xgb.train.data,    
                       nround = min_logloss_index, 
                       params = xgb_params)
  timeend <- Sys.time()
  runningtime <- timeend - timestart
  return(list(fit = xgb_fit, time = runningtime))
}

# test function
xgboost_test<- function(fit_model, testdata){
  # fit_model should be xgb$fit and testdata has to be a matrix
  xgb.test.data <- xgb.DMatrix(data = testdata)
  pred <- predict(fit_model,newdata = xgb.test.data) + 1 # predicted output of the xgboost is 0, 1 and 2.
  return(pred)
}
pred <- xgboost_test(xgb$fit,testdata) 
sum(pred==test.class[,2])/nrow(test.class)
