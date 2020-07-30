library(caTools) #to use sample.split function
library(caret) #to use train function
library(parallel) #to use parallel cores
library(GA)
library(xgboost)
library(precrec)
library(caret)
library(PRROC)





#read in csv file
cancer <- read.csv(file = "./Project/Machine Learning/Breast Cancer/breast-cancer.csv",encoding="UTF-8")

#show dimension
dim(cancer)

#show sample of dataset
head(cancer)

#show data type of each column
str(cancer)

#remove unnecessary column
cancer$X <- NULL
cancer$id <- NULL

#convert label to integer
cancer$diagnosis <- ifelse(cancer$diagnosis=="M",1,0)

#show summary of data
summary(cancer)

#check for NA value
sapply(cancer, function(x) sum(is.na(x)))

#scale the data
cancer[,c(2:31)] <- as.data.frame(apply(cancer[,c(2:31)], 2, function(x) scale(x)))

#check for proportion of class
prop.table(table(cancer$diagnosis))

#split the dataset into train and test subset
set.seed(1)
split = sample.split(cancer, SplitRatio = 0.7)
train = subset(cancer, split == TRUE)
test = subset(cancer, split == FALSE)

prop.table(table(train$diagnosis))
prop.table(table(test$diagnosis))

monitor <- function(obj){
  # gaMonitor(obj)                      #call the default gaMonitor to print the usual messages during evolution
  iter <- obj@iter                      #get the current iternation/generation number 
  if (iter <= max_iter){          #some error checking
    fitness <- obj@fitness              #get the array of all the fitness values in the present population
    #<<- assigns a value to the global variable declared outside the scope of this function.    
    thisRunResults[iter,1] <<- max(fitness)
    thisRunResults[iter,2] <<- mean(fitness)    
    thisRunResults[iter,3] <<- median(fitness)
    cat(paste("\rGA | iter =", obj@iter, "Mean =", thisRunResults[iter,2], "| Best =", thisRunResults[iter,1], "\n"))
    flush.console()
  }  
  else{                               #print error messages
    cat("ERROR: iter = ", iter, "exceeds maxGenerations = ", max_iter, ".\n")
    cat("Ensure maxGenerations == nrow(thisRunResults)")
  }
}

#create an fitness function for the XGBoost algorithm
eval_function_XGBoost <- function(x1, x2, x3, x4, x5, x6) {
  
  dtrain <- xgb.DMatrix(data = as.matrix(train[-1]), label = train$diagnosis)
  
  XGBoost_model <- xgb.cv(data = dtrain,
                          booster = "gbtree",
                          metric = "aucpr",
                          objective = "binary:logistic",
                          nfold = 5,
                          early_stopping_round = 30,
                          nrounds = 500,
                          max_depth = round(x1),
                          eta = x2,
                          gamma = x3,
                          colsample_bytree = x4,
                          min_child_weight = x5,
                          subsample = x6
                          )
  
  return(max(XGBoost_model$evaluation_log$test_aucpr_mean)) #maximize the Area Under the Precision Recal Curve
  
}

#set parameter setting for search algorithms
max_iter <<- 10 # maximum number of iterations
pop_size <- 50 # population size

# Define minimum and maximum values for each input
max_depth_min_max <- c(0,20)
eta_min_max <- c(0,1)
gamma_min_max <- c(0,20)
colsample_bytree_min_max <- c(0,1)
min_child_weight_min_max <- c(0,20)
subsample_min_max <- c(0,1)

runGA <- function(noRuns = 30){
  GA_T0 <- Sys.time()
  
  #Set up what stats you wish to note.    
  statnames = c("best", "mean", "median")
  thisRunResults <<- matrix(nrow=max_iter, ncol = length(statnames)) #stats of a single run
  resultsMatrix = matrix(1:max_iter, ncol = 1)  #stats of all the runs
  
  resultNames = character(length(statnames)*noRuns)
  resultNames[1] = "Generation"
  
  bestFitness <<- -Inf
  bestSolution <<- NULL
  
  seed <- sample(1:1000, noRuns, replace=F)
  
  for (i in 1:noRuns){
    cat(paste("Starting Run ", i, "\n"))
    # Run genetic algorithm
    GA <- ga(type = "real-valued", fitness = function(x) eval_function_XGBoost(x[1],x[2],x[3],x[4],x[5],x[6]), 
             lower = c( max_depth_min_max[1], eta_min_max[1], gamma_min_max[1], colsample_bytree_min_max[1], min_child_weight_min_max[1], subsample_min_max[1]), # minimum values
             upper = c(max_depth_min_max[2], eta_min_max[2], gamma_min_max[2], colsample_bytree_min_max[2], min_child_weight_min_max[2], subsample_min_max[2]), # maximum values
             popSize = pop_size, # population size
             maxiter = max_iter, # number of iterations
             pcrossover = 0.8, # probability of crossover
             pmutation = 0.2, # probability of mutation
             elitism = 5, # number of best current solutions used on next round
             parallel = T,
             optim = F, 
             keepBest = T,
             seed = seed[i],
             monitor = monitor)
    
    GA_T1 <- Sys.time()
    
    cat(paste("Processing Time: ", GA_T1 - GA_T0, "\n"))
    
    resultsMatrix = cbind(resultsMatrix, thisRunResults)
    
    if (GA@fitnessValue > bestFitness){
      bestFitness <<- GA@fitnessValue
      bestSolution <<- GA@solution
    }
    
    #Create column names for the resultsMatrix
    for (j in 1:length(statnames)) resultNames[1+(i-1)*length(statnames)+j] = paste(statnames[j],i)
  }
  
  colnames(resultsMatrix) = resultNames
  return (resultsMatrix)
}

getBestFitness<-function(){
  return(bestFitness)
}

getBestSolution<-function(){
  return(bestSolution)
}

dtrain <- xgb.DMatrix(data = as.matrix(train[-1]), label = train$diagnosis)

seed <- sample(1:1000, 30, replace=F)

AUC_ROC = c()
AUC_PR = c()
AUC_Recall = c()
AUC_Accuracy = c()
AUC_LogLoss = c()
for (i in 1:30){
  set.seed(seed[i])
  XGBoost_model <- xgb.train(data = dtrain,
                           nrounds = 200,
                           max_depth = round(aucSolution[1]),
                           eta = aucSolution[2],
                           gamma = aucSolution[3],
                           colsample_bytree = aucSolution[4],
                           min_child_weight = aucSolution[5],
                           subsample = aucSolution[6],
                           booster = "gbtree",
                           metric = "auc",
                           objective = "binary:logistic")
  
  y_pred <- predict(XGBoost_model, data.matrix(test[,-1]))
  prediction <- as.numeric(y_pred > 0.5)
  
  fg <- y_pred[test$diagnosis == 1]
  bg <- y_pred[test$diagnosis == 0]
  
  roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  roc
  
  pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  pr
  
  AUC_ROC[i] <- roc$auc
  AUC_PR[i] <- pr$auc.integral
  AUC_Recall[i] <- Recall(test$diagnosis, prediction)
  AUC_Accuracy[i] <- Accuracy(test$diagnosis, prediction)
  AUC_LogLoss[i] <- LogLoss(y_pred, test$diagnosis)
}

AUCPR_ROC = c()
AUCPR_PR = c()
AUCPR_Recall = c()
AUCPR_Accuracy = c()
AUCPR_LogLoss = c()
for (i in 1:30){
  set.seed(1)
  XGBoost_model <- xgb.train(data = dtrain,
                           nrounds = 200,
                           max_depth = round(aucprSolution[1]),
                           eta = aucprSolution[2],
                           gamma = aucprSolution[3],
                           colsample_bytree = aucprSolution[4],
                           min_child_weight = aucprSolution[5],
                           subsample = aucprSolution[6],
                           booster = "gbtree",
                           metric = "aucpr",
                           objective = "binary:logistic")

  y_pred <- predict(XGBoost_model, data.matrix(test[,-1]))
  prediction <- as.numeric(y_pred > 0.5)
  
  fg <- y_pred[test$diagnosis == 1]
  bg <- y_pred[test$diagnosis == 0]
  
  roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  roc
  
  pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  pr
  
  AUCPR_ROC[i] <- roc$auc
  AUCPR_PR[i] <- pr$auc.integral
  AUCPR_Recall[i] <- Recall(test$diagnosis, prediction)
  AUCPR_Accuracy[i] <- Accuracy(test$diagnosis, prediction)
  AUCPR_LogLoss[i] <- LogLoss(y_pred, test$diagnosis)
  
}

LogLoss_ROC = c()
LogLoss_PR = c()
LogLoss_Recall = c()
LogLoss_Accuracy = c()
LogLoss_LogLoss = c()
for (i in 1:30){
  set.seed(seed[i])
  XGBoost_model <- xgb.train(data = dtrain,
                             nrounds = 200,
                             max_depth = round(loglossSolution[1]),
                             eta = loglossSolution[2],
                             gamma = loglossSolution[3],
                             colsample_bytree = loglossSolution[4],
                             min_child_weight = loglossSolution[5],
                             subsample = loglossSolution[6],
                             booster = "gbtree",
                             metric = "aucpr",
                             objective = "binary:logistic")
  
  y_pred <- predict(XGBoost_model, data.matrix(test[,-1]))
  prediction <- as.numeric(y_pred > 0.5)
  
  fg <- y_pred[test$diagnosis == 1]
  bg <- y_pred[test$diagnosis == 0]
  
  roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  roc
  
  pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  pr
  
  LogLoss_ROC[i] <- roc$auc
  LogLoss_PR[i] <- pr$auc.integral
  LogLoss_Recall[i] <- Recall(test$diagnosis, prediction)
  LogLoss_Accuracy[i] <- Accuracy(test$diagnosis, prediction)
  LogLoss_LogLoss[i] <- LogLoss(y_pred, test$diagnosis)
}

confusionMatrix(as.factor(prediction), as.factor(test$diagnosis), positive = "1", mode = "prec_recall")

AUC_data <- data.frame( 
  group = rep(c("LogLoss", "AUCPR", "AUC"), each = 30),
  ROC = c(LogLoss_ROC,  AUCPR_ROC, AUC_ROC)
)

PR_data <- data.frame( 
  group = rep(c("LogLoss", "AUCPR", "AUC"), each = 30),
  AUCPR = c(LogLoss_PR,  AUCPR_PR, AUC_PR)
)

Recall_data <- data.frame( 
  group = rep(c("LogLoss", "AUCPR", "AUC"), each = 30),
  Recall = c(LogLoss_Recall,  AUCPR_Recall, AUC_Recall)
)

Accuracy_data <- data.frame( 
  group = rep(c("LogLoss", "AUCPR", "AUC"), each = 30),
  Accuracy = c(LogLoss_Accuracy,  AUCPR_Accuracy, AUC_Accuracy)
)

LogLoss_data <- data.frame( 
  group = rep(c("LogLoss", "AUCPR", "AUC"), each = 30),
  LogLoss = c(LogLoss_LogLoss,  AUCPR_LogLoss, AUC_LogLoss)
)


library("ggpubr")
ggboxplot(Accuracy_data, x = "group", y = "Accuracy", 
          color = "group", palette = c("#FF0000", "#008000", "#0000FF"),
          ylab = "Accuracy", xlab = "Metric")

AUC(y_pred, test$diagnosis)
