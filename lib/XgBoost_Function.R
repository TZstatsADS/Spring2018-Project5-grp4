train_xgboost <- function(traindata,nround = 200, cv.nfold = 5){
  library("xgboost")
  # traindata has to be a matrix
  timestart <- Sys.time()
  # Data Preparation
  processed.data <- apply(traindata[,-c(7,8)], 2, as.numeric)
  train.data     <- as.matrix(processed.data[,c("ip","app","device","os","channel","Hour")])
  train.class    <- as.matrix(processed.data[,"is_attributed"])
  xgb.train.data <- xgb.DMatrix(data = train.data,label = train.class)
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

