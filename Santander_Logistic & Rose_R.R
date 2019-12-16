#clear the environment
rm(list=ls())
#load required libraries
x=c("tidyverse","moments","DataExplorer","caret","Matrix","mlbench","caTools","randomForest","glmnet","mlr","unbalanced", "vita","rBayesianOptimization","xgboost","boot","pROC","DMwR","ROSE","yardstick")
lapply(x,require,character.only=TRUE)
rm(x)
#setting working directory
setwd("F:/EdwisorVanusha/Project/Santander")
#loading the train data
train=read.csv("train.csv")
test=read.csv("test.csv")
#########################################Exploratory Data Analysis####################################################
#Let's check the dimension of train and test sets. Also check what are the variables that are there in train but not in test. Also let's have a look at the head of the data sets
dim(train) ; dim(test) ; setdiff(colnames(train) , colnames(test)) ; head(train) ; head(test)
#It seems like the variables have no names as such and the only variable that is missing in the test set is the target column which we need to predict.
#Summary of the dataset
str(train)
#Typecasting the target variable
#convert to factor
train$target<-as.factor(train$target)
#Target classes count in train data
require(gridExtra)
#Count of target classes
table(train$target)
#Percenatge counts of target classes
table(train$target)/length(train$target)*100
#################################################Data Visualisation##################################################
#Bar plot for count of target classes
plot1<-ggplot(train,aes(target))+theme_bw()+geom_bar(stat='count',fill='lightgreen')
#Violin with jitter plots for target classes
plot2<-ggplot(train,aes(x=target,y=1:nrow(train)))+theme_bw()+geom_violin(fill='lightblue')+
  facet_grid(train$target)+geom_jitter(width=0.02)+labs(y='Index')
grid.arrange(plot1,plot2, ncol=2)
#Distribution of train attributes from 3 to 102
for (var in names(train)[c(3:102)]){
  target<-train$target
  plot<-ggplot(train, aes(x=train[[var]],fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot)
}
#Distribution of train attributes from 103 to 202
for (var in names(train)[c(103:202)]){
  target<-train$target
  plot<-ggplot(train, aes(x=train[[var]], fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot)
}
#Distribution of test attributes from 2 to 101
plot_density(test[,c(2:101)], ggtheme = theme_classic(),geom_density_args = list(color='blue'))
#Distribution of test attributes from 102 to 201
plot_density(test[,c(102:201)], ggtheme = theme_classic(),geom_density_args = list(color='blue'))
#Applying the function to find mean values per row in train and test data.
train_mean<-apply(train[,-c(1,2)],MARGIN=1,FUN=mean)
test_mean<-apply(test[,-c(1)],MARGIN=1,FUN=mean)
ggplot()+
  #Distribution of mean values per row in train data
  geom_density(data=train[,-c(1,2)],aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per row in test data
  geom_density(data=test[,-c(1)],aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per row',title="Distribution of mean values per row in train and test dataset")

#Applying the function to find mean values per column in train and test data.
train_mean<-apply(train[,-c(1,2)],MARGIN=2,FUN=mean)
test_mean<-apply(test[,-c(1)],MARGIN=2,FUN=mean)
ggplot()+
  #Distribution of mean values per column in train data
  geom_density(aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per column in test data
  geom_density(aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per column',title="Distribution of mean values per row in train and test dataset")
#Applying the function to find mean values per column in train and test data.
train_mean<-apply(train[,-c(1,2)],MARGIN=2,FUN=mean)
test_mean<-apply(test[,-c(1)],MARGIN=2,FUN=mean)
ggplot()+
  #Distribution of mean values per column in train data
  geom_density(aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per column in test data
  geom_density(aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per column',title="Distribution of mean values per row in train and test dataset")

#Let us see distribution of standard deviation values per row and column in train and test dataset
#Applying the function to find standard deviation values per row in train and test data.
train_sd<-apply(train[,-c(1,2)],MARGIN=1,FUN=sd)
test_sd<-apply(test[,-c(1)],MARGIN=1,FUN=sd)
ggplot()+
  #Distribution of sd values per row in train data
  geom_density(data=train[,-c(1,2)],aes(x=train_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
  #Distribution of mean values per row in test data
  geom_density(data=test[,-c(1)],aes(x=test_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='sd values per row',title="Distribution of sd values per row in train and test dataset")

#Applying the function to find sd values per column in train and test data.
train_sd<-apply(train[,-c(1,2)],MARGIN=2,FUN=sd)
test_sd<-apply(test[,-c(1)],MARGIN=2,FUN=sd)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='sd values per column',title="Distribution of std values per column in train and test dataset")
#Let us see distribution of skewness values per row and column in train and test dataset
#Applying the function to find skewness values per row in train and test data.
train_skew<-apply(train[,-c(1,2)],MARGIN=1,FUN=skewness)
test_skew<-apply(test[,-c(1)],MARGIN=1,FUN=skewness)
ggplot()+
  #Distribution of skewness values per row in train data
  geom_density(aes(x=train_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
  #Distribution of skewness values per column in test data
  geom_density(aes(x=test_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='skewness values per row',title="Distribution of skewness values per row in train and test dataset")

#Applying the function to find skewness values per column in train and test data.
train_skew<-apply(train[,-c(1,2)],MARGIN=2,FUN=skewness)
test_skew<-apply(test[,-c(1)],MARGIN=2,FUN=skewness)
ggplot()+
  #Distribution of skewness values per column in train data
  geom_density(aes(x=train_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
  #Distribution of skewness values per column in test data
  geom_density(aes(x=test_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='skewness values per column',title="Distribution of skewness values per column in train and test dataset")
#Let us see distribution of kurtosis values per row and column in train and test dataset
#Applying the function to find kurtosis values per row in train and test data.
train_kurtosis<-apply(train[,-c(1,2)],MARGIN=1,FUN=kurtosis)
test_kurtosis<-apply(test[,-c(1)],MARGIN=1,FUN=kurtosis)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
  labs(x='kurtosis values per row',title="Distribution of kurtosis values per row in train and test dataset")

#Applying the function to find kurtosis values per column in train and test data.
train_kurtosis<-apply(train[,-c(1,2)],MARGIN=2,FUN=kurtosis)
test_kurtosis<-apply(test[,-c(1)],MARGIN=2,FUN=kurtosis)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
  labs(x='kurtosis values per column',title="Distribution of kurtosis values per column in train and test dataset")

###########################################Missing value analysis###########################################################
#Let us do Missing value analysis
#Finding the missing values in train data
missing_val<-data.frame(missing_val=apply(train,2,function(x){sum(is.na(x))}))
missing_val<-sum(missing_val)
missing_val
#Finding the missing values in test data
missing_val<-data.frame(missing_val=apply(test,2,function(x){sum(is.na(x))}))
missing_val<-sum(missing_val)
missing_val
#No missing values are present in both train and test data.
#############################################Outlier analysis###################################################
# No need of doing outlier analysis on imbalanced data as minority class observations will be shown as outliers and we cant remove them.
##########################################################Feature Selection#################################
##############################################Correlations in train data############################################
#convert factor to int
train$target<-as.numeric(train$target)
train_correlations<-cor(train[,c(2:202)])
train_correlations
#We can observed that the correlation between the train attributes is very small.
#Correlations in test data
test_correlations<-cor(test[,c(2:201)])
test_correlations
#We can observed that the correlation between the test attributes is very small.
#################################################Feature Engineering#################################################
################################################################Variable importance##########################################
#Split the training data using simple random sampling
train_index<-sample(1:nrow(train),0.75*nrow(train))
#train data
train_data<-train[train_index,]
#validation data
valid_data<-train[-train_index,]
#dimension of train and validation data
dim(train_data)
dim(valid_data)
#Training the Random forest classifier
set.seed(2732)
#convert to int to factor
train_data$target<-as.factor(train_data$target)
#setting the mtry
mtry<-floor(sqrt(200))
#setting the tunegrid
tuneGrid<-expand.grid(.mtry=mtry)
#fitting the ranndom forest
rf<-randomForest(target~.,train_data[,-c(1)],mtry=mtry,ntree=10,importance=TRUE)
#Variable importance
VarImp<-importance(rf,type=2)
VarImp
#We can observed that the top important features are var_12, var_26, var_22,v var_174, var_198 and so on based on Mean decrease gini.
####################################################Model development#####################################################
#Split the data using CreateDataPartition
set.seed(689)
#train.index<-createDataPartition(train$target,p=0.8,list=FALSE)
train.index<-sample(1:nrow(train),0.8*nrow(train))
#train data
train.data<-train[train.index,]
#validation data
valid.data<-train[-train.index,]
#dimension of train data
dim(train.data)
#dimension of validation data
dim(valid.data)
#target classes in train data
table(train.data$target)
#target classes in validation data
table(valid.data$target)
#Logistic Regression model
#Training and validation dataset
#Training dataset
X_t<-as.matrix(train.data[,-c(1,2)])
y_t<-as.matrix(train.data$target)
#validation dataset
X_v<-as.matrix(valid.data[,-c(1,2)])
y_v<-as.matrix(valid.data$target)
#test dataset
test<-as.matrix(test[,-c(1)])
#Logistic regression model
set.seed(667) # to reproduce results
lr_model <-glmnet(X_t,y_t, family = "binomial")
summary(lr_model)
#Cross validation prediction
set.seed(8909)
cv_lr <- cv.glmnet(X_t,y_t,family = "binomial", type.measure = "class")
cv_lr
#Plotting the missclassification error vs log(lambda) where lambda is regularization parameter
#Minimum lambda
cv_lr$lambda.min
#plot the auc score vs log(lambda)
plot(cv_lr)
#Model performance on validation dataset
set.seed(5363)
cv_predict.lr<-predict(cv_lr,X_v,s = "lambda.min", type = "class")
cv_predict.lr
#Accuracy of the model is not the best metric to use when evaluating the imbalanced datasets as it may be misleading. So, we are going to change the performance metric.
#Confusion matrix
set.seed(689)
#actual target variable
target<-valid.data$target
#convert to factor
target<-as.factor(target)
#predicted target variable
#convert to factor
cv_predict.lr<-as.factor(cv_predict.lr)
confusionMatrix(data=cv_predict.lr,reference=target)
#ROC_AUC score and curve
set.seed(892)
cv_predict.lr<-as.numeric(cv_predict.lr)
valid.data
response=valid.data$target
roc(data=valid.data [,-c(1,2)],response=target,predictor=cv_predict.lr,auc=TRUE,plot=TRUE)
#When we compare the model accuracy and roc_auc score, conclude that the model is not performing well on imbalanced data.
#Random Oversampling Examples(ROSE)
#It creates a sample of synthetic data by enlarging the features space of minority and majority class examples.
set.seed(699)
train.rose <- ROSE(target~., data =train.data[,-c(1)],seed=32)$data
#target classes in balanced train data
table(train.rose$target)
valid.rose <- ROSE(target~., data =valid.data[,-c(1)],seed=42)$data
#target classes in balanced valid data
table(valid.rose$target)
#Let us see how baseline logistic regression model performs on synthetic data points.
#Logistic regression model
set.seed(462)
lr_rose <-glmnet(as.matrix(train.rose),as.matrix(train.rose$target), family = "binomial")
summary(lr_rose)
#Cross validation prediction
set.seed(473)
cv_rose = cv.glmnet(as.matrix(valid.rose),as.matrix(valid.rose$target),family = "binomial", type.measure = "class")
cv_rose
#Minimum lambda
cv_rose$lambda.min
#plot the auc score vs log(lambda)
plot(cv_rose)
#Model performance on validation dataset
set.seed(442)
cv_predict.rose<-predict(cv_rose,as.matrix(valid.rose),s = "lambda.min", type = "class")
cv_predict.rose
#Confusion matrix
set.seed(478)
#actual target variable
target<-valid.rose$target
#convert to factor
target<-as.factor(target)
#predicted target variable
#convert to factor
cv_predict.rose<-as.factor(cv_predict.rose)
#Confusion matrix
confusionMatrix(data=cv_predict.rose,reference=target)
#ROC_AUC score and curve
set.seed(843)
#convert to numeric
cv_predict.rose<-as.numeric(cv_predict.rose)
roc(data=valid.rose[,-c(1,2)],response=target,predictor=cv_predict.rose,auc=TRUE,plot=TRUE)
#We can observed that ROSE model is performing well on imbalance data compare to baseline logistic regression.
#XG Boost is in next R file