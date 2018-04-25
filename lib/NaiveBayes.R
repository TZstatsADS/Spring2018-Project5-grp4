NaiveBayes <- function(TRAIN = "traindata.csv", TEST = "testdata.csv",IP=1, APP=1, OS=1, DV=1, CH=1, TM=0)
{
  TimeToNum <- function(x)
  {
    Hour <- as.numeric(substr(x, start = 12, stop = 13)) * 2
    HourHalf <- as.numeric(substr(x, start = 15, stop = 15))
    if (HourHalf >= 3)
      Hour <- Hour + 1
    return(Hour)
  }
  CalcProb <- function(User, deno)
  {
    if (is.na(tmp[User]))
      return(lambda / deno)
    return((tmp[User] + lambda) / deno)
  }
  
  TrainData <- read.csv(file = TRAIN)
  TrainData$Hour <- mapply(TimeToNum, TrainData)
  #head(TrainData)
  #table(TrainData$is_attributed)
  
  IpCount <- table(TrainData$ip)
  AppCount <- table(TrainData$app)
  OsCount <- table(TrainData$os)
  DvCount <- table(TrainData$device)#TrainData$device: 0,1,2,3(Others)
  ChCount <- table(TrainData$channel)
  TmCount <- table(TrainData$Hour)
  
  TrainData$ip <- as.character(TrainData$ip)
  TrainData$app <- as.character(TrainData$app)
  TrainData$os <- as.character(TrainData$os)
  TrainData$device <- as.character(TrainData$device)
  TrainData$channel <- as.character(TrainData$channel)
  TrainData$is_attributed <- as.numeric(TrainData$is_attributed) + 1
  
  lambda <- 0.1
  Prior <- rep(NA, 2)
  Prior[1] <- (sum(1 - TrainData$is_attributed) + lambda) / (nrow(TrainData) + lambda * 2)
  Prior[2] <- (sum(TrainData$is_attributed) + lambda) / (nrow(TrainData) + lambda * 2)
  
  #calc IP Prob
  if (IP==1){
    IpTest <-  names(table(TestData$ip))
    IpTestNew <- setdiff(IpTest, names(IpCount))
    IpIntersect <- intersect(IpTest, names(IpCount))
    #IpProbNew <- matrix(0, nrow = length(IpTestNew), ncol = 2)
    
    IpProb <- matrix(0, nrow = length(IpIntersect) + length(IpTestNew), ncol = 2)
    row.names(IpProb) <- c(IpIntersect, IpTestNew)
    
    tmp <- table(TrainData$ip[which(TrainData$is_attributed == 0)])
    deno <- (sum(1 - TrainData$is_attributed) + length(IpCount) * lambda)
    IpProb[,1] <- mapply(CalcProb, row.names(IpProb), deno = deno)
    
    tmp <- table(TrainData$ip[which(TrainData$is_attributed == 1)])
    deno <- (sum(TrainData$is_attributed) + length(IpCount) * lambda)
    IpProb[,2] <- mapply(CalcProb, row.names(IpProb), deno = deno)
  }
  #calc App prob
  if (APP==1){
    AppTest <-  names(table(TestData$app))
    AppTestNew <- setdiff(AppTest, names(AppCount))
    AppIntersect <- intersect(AppTest, names(AppCount))
    #AppProbNew <- matrix(0, nrow = length(AppTestNew), ncol = 2)
    AppProb <- matrix(0, nrow = length(AppCount) + length(AppTestNew), ncol = 2)
    row.names(AppProb) <- c(names(AppCount), AppTestNew)
    
    tmp <- table(TrainData$app[which(TrainData$is_attributed == 0)])
    deno <- (sum(1 - TrainData$is_attributed) + length(AppCount) * lambda)
    AppProb[,1] <- mapply(CalcProb, row.names(AppProb), deno = deno)
    tmp <- table(TrainData$app[which(TrainData$is_attributed == 1)])
    deno <- (sum(TrainData$is_attributed) + length(AppCount) * lambda)
    AppProb[,2] <- mapply(CalcProb, row.names(AppProb), deno = deno)
  }
  #calc Os prob
  if (OS==1){
    OsTest <-  names(table(TestData$os))
    OsTestNew <- setdiff(OsTest, names(OsCount))
    OsIntersect <- intersect(OsTest, names(OsCount))
    #AppProbNew <- matrix(0, nrow = length(AppTestNew), ncol = 2)
    OsProb <- matrix(0, nrow = length(OsCount) + length(OsTestNew), ncol = 2)
    row.names(OsProb) <- c(names(OsCount), OsTestNew)
    
    tmp <- table(TrainData$os[which(TrainData$is_attributed == 0)])
    deno <- (sum(1 - TrainData$is_attributed) + length(OsCount) * lambda)
    OsProb[,1] <- mapply(CalcProb, row.names(OsProb), deno = deno)
    tmp <- table(TrainData$os[which(TrainData$is_attributed == 1)])
    deno <- (sum(TrainData$is_attributed) + length(OsCount) * lambda)
    OsProb[,2] <- mapply(CalcProb, row.names(OsProb), deno = deno)
  }
  #calc Ch prob
  if (CH==1){
    ChTest <-  names(table(TestData$channel))
    ChTestNew <- setdiff(ChTest, names(ChCount))
    ChIntersect <- intersect(ChTest, names(ChCount))
    #ChProbNew <- matrix(0, nrow = length(ChTestNew), ncol = 2)
    ChProb <- matrix(0, nrow = length(ChCount) + length(ChTestNew), ncol = 2)
    row.names(ChProb) <- c(names(ChCount), ChTestNew)
    
    tmp <- table(TrainData$channel[which(TrainData$is_attributed == 0)])
    deno <- (sum(1 - TrainData$is_attributed) + length(ChCount) * lambda)
    ChProb[,1] <- mapply(CalcProb, row.names(ChProb), deno = deno)
    tmp <- table(TrainData$channel[which(TrainData$is_attributed == 1)])
    deno <- (sum(TrainData$is_attributed) + length(ChCount) * lambda)
    ChProb[,2] <- mapply(CalcProb, row.names(ChProb), deno = deno)
  }
  #calc device prob
  if (DV==1){
    DvTest <-  names(table(TestData$device))
    DvTestNew <- setdiff(DvTest, names(DvCount))
    DvIntersect <- intersect(DvTest, names(DvCount))
    #AppProbNew <- matrix(0, nrow = length(AppTestNew), ncol = 2)
    DvProb <- matrix(0, nrow = length(DvCount) + length(DvTestNew), ncol = 2)
    row.names(DvProb) <- c(names(DvCount), DvTestNew)
    
    tmp <- table(TrainData$device[which(TrainData$is_attributed == 0)])
    deno <- (sum(1 - TrainData$is_attributed) + length(DvCount) * lambda)
    DvProb[,1] <- mapply(CalcProb, row.names(DvProb), deno = deno)
    tmp <- table(TrainData$device[which(TrainData$is_attributed == 1)])
    deno <- (sum(TrainData$is_attributed) + length(DvCount) * lambda)
    DvProb[,2] <- mapply(CalcProb, row.names(DvProb), deno = deno)
  }
  #calc Time prob
  if (TM==1){
    TmTest <-  names(table(TestData$Hour))
    TmTestNew <- setdiff(TmTest, names(TmCount))
    TmIntersect <- intersect(TmTest, names(TmCount))
    #AppProbNew <- matrix(0, nrow = length(AppTestNew), ncol = 2)
    TmProb <- matrix(0, nrow = length(TmCount) + length(TmTestNew), ncol = 2)
    row.names(TmProb) <- c(names(TmCount), TmTestNew)
    
    tmp <- table(TrainData$Hour[which(TrainData$is_attributed == 0)])
    deno <- (sum(1 - TrainData$is_attributed) + length(TmCount) * lambda)
    TmProb[,1] <- mapply(CalcProb, row.names(TmProb), deno = deno)
    tmp <- table(TrainData$Hour[which(TrainData$is_attributed == 1)])
    deno <- (sum(TrainData$is_attributed) + length(TmCount) * lambda)
    TmProb[,2] <- mapply(CalcProb, row.names(TmProb), deno = deno)
  }
  
  TestData <- read.csv(TEST)
  
  TestData$Prob0 <- rep(Prior[1], nrow(TestData))
  TestData$Prob1 <- rep(Prior[2], nrow(TestData))
  TestData$ip <- as.character(TestData$ip)
  TestData$app <- as.character(TestData$app)
  TestData$os <- as.character(TestData$os)
  TestData$device <- as.character(TestData$device)
  TestData$channel <- as.character(TestData$channel)
  TestData$Hour <- as.character(TestData$Hour)
  
  if (IP==1)
  {
    TestData$Prob0 <- TestData$Prob0 * IpProb[TestData$ip, 1] 
    TestData$Prob1 <- TestData$Prob1 * IpProb[TestData$ip, 2] 
  }
  if (APP==1)
  {
    TestData$Prob0 <- TestData$Prob0 * AppProb[TestData$app, 1] 
    TestData$Prob1 <- TestData$Prob1 * AppProb[TestData$app, 2] 
  }
  if (OS==1)
  {
    TestData$Prob0 <- TestData$Prob0 * OsProb[TestData$os, 1] 
    TestData$Prob1 <- TestData$Prob1 * OsProb[TestData$os, 2] 
  }
  if (DV==1)
  {
    TestData$Prob0 <- TestData$Prob0 * DvProb[TestData$device, 1] 
    TestData$Prob1 <- TestData$Prob1 * DvProb[TestData$device, 2] 
  }
  if (CH==1)
  {
    TestData$Prob0 <- TestData$Prob0 * ChProb[TestData$channel, 1] 
    TestData$Prob1 <- TestData$Prob1 * ChProb[TestData$channel, 2] 
  }
  if (TM==1)
  {
    TestData$Prob0 <- TestData$Prob0 * TmProb[TestData$Hour, 1] 
    TestData$Prob1 <- TestData$Prob1 * TmProb[TestData$Hour, 2] 
  }
  TestData$Ans <- round(TestData$Prob1 / (TestData$Prob0 + TestData$Prob1), 7)
  
  Ans <- TestData[,c("click_id", "Ans")]
  names(Ans)[2] <- "is_attributed"
  write.csv(Ans, file = "Ans.csv",row.names = FALSE)
}