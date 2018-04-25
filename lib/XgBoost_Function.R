train_xgboost <- function(traindata){
  # traindata has to be a matrix
  timestart <- Sys.time()
  # Data Preparation
  xgb.train.data <- xgb.DMatrix(data = traindata[,-1],label = traindata[,1] - 1)
  # Default Parameter 
  xgb_params <- list("objective" = "binary:logistic", 
                     "eval_metric" = "auc",          
                     "silent"="0",
                     "booster" = "gbtree")
  # Nrounds in the XgBoost
  cv_model <- xgb.cv(params      = xgb_params,
                     data        = xgb.train.data, 
                     nrounds     = nround,
                     nfold       = cv.nfold,
                     verbose     = TRUE,
                     prediction  = TRUE,
                     tree_method = 'exact')
  
  max_auc = max(cv_model[["evaluation_log"]][, 4])    
  max_auc_index = max((1:nround)[cv_model[["evaluation_log"]][, 4] == max_auc]) 
  
  xgb_fit <- xgb.train(data = xgb.train.data,    
                       nround = max_auc_index, 
                       params = xgb_params,
                       tree_method = 'exact')
  
  timeend <- Sys.time()
  runningtime <- timeend - timestart
  return(list(fit = xgb_fit, time = runningtime))
}