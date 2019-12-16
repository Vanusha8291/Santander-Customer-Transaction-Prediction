library(data.table)
library(caret)
library(xgboost)
library(pROC)
#setting working directory
setwd("F:/EdwisorVanusha/Project/Santander")
#loading the train data
train=read.csv("train.csv")
test=read.csv("test.csv")
#Let's check the dimension of train and test sets. Also check what are the variables that are there in train but not in test. Also let's have a look at the head of the data sets
dim(train) ; dim(test) ; setdiff(colnames(train) , colnames(test)) ; head(train) ; head(test)
#It seems like the variables have no names as such and the only variable that is missing in the test set is the target column which we need to predict. Let's check the sample submission file. We will check if the ids in sample sub and the test file are in same order or not as well.
#training and testing data
trainX=as.matrix(train[-c(1,2)])
trainY=as.matrix(train$target)
testX=as.matrix(test[-c(1)])
# preparing XGB matrix
dtrain <- xgb.DMatrix(data = as.matrix(trainX), label = as.matrix(trainY))
# parameters
params <- list(booster = "gbtree",
               objective = "binary:logistic",
               eta=0.02,
               #gamma=80,
               max_depth=2,
               min_child_weight=1, 
               subsample=0.5,
               colsample_bytree=0.1,
               scale_pos_weight = round(sum(!trainY) / sum(trainY), 2))
# CV
set.seed(123)
xgbcv <- xgb.cv(params = params, 
                data = dtrain, 
                nrounds = 30000, 
                nfold = 5,
                showsd = F, 
                stratified = T, 
                print_every_n = 100, 
                early_stopping_rounds = 500, 
                maximize = T,
                metrics = "auc")
cat(paste("Best iteration:", xgbcv$best_iteration))
# train final model
set.seed(123)
xgb_model <- xgb.train(
  params = params, 
  data = dtrain, 
  nrounds = xgbcv$best_iteration, 
  print_every_n = 100, 
  maximize = T,
  eval_metric = "auc")
#view variable importance plot
imp_mat <- xgb.importance(feature_names = colnames(trainX), model = xgb_model)
xgb.plot.importance(importance_matrix = imp_mat[1:30])
#prediction
pred_sub <- predict(xgb_model, newdata=as.matrix(testX), type="response")
#Submission
submission <- read.csv("submission.csv")
submission$target <- pred_sub
write.csv(submission, file="submission_XgBoost.csv", row.names=F)
