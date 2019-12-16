#!/usr/bin/env python
# coding: utf-8

# # Prediction of Santander Customer Transaction
# 

# #### Problem Statement:
# We need to identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted.
# - Classification: The target variable is a binary variable, 0 (will make a specific transaction in the future), 1 (will not make a specific transaction in the future)
Contents:
1.Exploratory Data Analysis
   * Loading dataset and libraries
   * Profiling the "train" dataset
   * Typecasting the attributes
   * Distribution of Target Class
   Attributes Distributions and trends
   * Distribution of train attributes
   * Distribution of test attributes
   * Mean distribution of attributes
   * Standard deviation distribution of attributes
   * Skewness distribution of attributes
   * Kurtosis distribution of attributes       
   * Missing value analysis
   * Outliers analysis     
2. Correlation matrix
3. Feature Engineering
   * Permutation Importance
3. Split the dataset into train and test dataset
4. Modelling the training dataset
   * Logistic Regression Model
   * SMOTE Model
   * LightGBM Model
5. Cross Validation Prediction
   * Logistic  Regression CV Prediction
   * SMOTE CV Prediction
   * LightGBM CV Prediction
6. Model performance on test dataset
   * Logistic Regression Prediction
   * SMOTE Prediction 
   * LightGBM Prediction
7. Model Evaluation Metrics
   * Confusion Matrix
   * ROC_AUC score
8. Choosing best model for predicting customer transaction
 Model Explantion using Partial Dependency Plot 
# #### Exploratory Data Analysis (EDA)
# 

# ##### Loading Libraries

# In[1]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from scipy import stats
import seaborn as sns
import missingno as msno
import pandas_profiling as pp
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

# import the necessary modelling algos.

#classification.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import eli5
from eli5.sklearn import PermutationImportance
from sklearn import tree
import graphviz
from pdpbox import pdp, get_dataset, info_plots

from scipy.stats import randint as sp_randint
import warnings
warnings.filterwarnings('ignore')

#model selection
from sklearn.model_selection import train_test_split,cross_validate,cross_val_score,cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

#Imbalanced data handling
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids,NearMiss, RandomUnderSampler

#evaluation metrics
from sklearn.metrics import roc_auc_score,confusion_matrix,make_scorer,classification_report,roc_curve,auc
get_ipython().system(' pip install scikit-plot')
import scikitplot as skplt
from scikitplot.metrics import plot_confusion_matrix,plot_precision_recall_curve


random_state=101
np.random.seed(random_state)


# ### Loading dataset

# In[2]:


os.chdir("F:\EdwisorVanusha\Project\Santander")
os.getcwd()


# In[3]:


train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")


# ##### Pandas_profiling
# Pandas Profiling is python package whichis a simple and fast way to perform exploratory data analysis of a Pandas Dataframe.
# Essentials: type, unique values, missing values, Quantile statistics like minimum value, Q1, median, Q3, maximum, range, interquartile range’ Descriptive statistics like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness, Most frequent values, Histogram, Correlations highlighting of highly correlated variables, Spearman, Pearson and Kendall matrices, Missing values matrix, count, heatmap and dendrogram of missing values

# In[4]:


#view profile report generated in the saved as a html file
#pfr.to_file("profile.html")


# In[5]:


#Shape of the train dataset
train.shape


# In[6]:


#Shape of the test dataset
test.shape


# In[7]:


#Summary of the dataset
train.describe()


# In[8]:


test.describe()


# ### Class distribution

# In[9]:


#target classes count
target_class=train['target'].value_counts()
print('Count of target classes :\n',target_class)
#Percentage of target classes count
per_target_class=train['target'].value_counts()/len(train)*100
print('percentage of count of target classes :\n',per_target_class)

##Count and pie chart to visualise 'target' class
f,ax=plt.subplots(1,2,figsize=(15,8))

train['target'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.2f%%',ax=ax[0],shadow=True)
ax[0].set_title('Target Distribution')
ax[0].set_ylabel('')
sns.countplot('target',data=train,ax=ax[1])
plt.show()


# Findings:
# Target variable has imbalanced class distribution where the number of observations belonging to one class is significantly lower than those belonging to the other class. Above visualisation shows that 89.95% of customers will not make the transaction (belong to "0") and 10.05% of customers make the transaction (belong to "1") at Santander

# ### Distributions of predictor variables

# Let's look at the predictor variables and see how the variables are distributed.

# ###### Predictor variables in train dataset 

# In[10]:


#Distribution of Independent Variables
def plot_train_attribute_distribution(t0,t1,label1,label2,train_attributes):
    i=0
    sns.set_style('whitegrid')
    
    fig=plt.figure()
    ax=plt.subplots(10,10,figsize=(22,18))
    
    for attribute in train_attributes:
        i+=1
        plt.subplot(10,10,i)
        sns.distplot(t0[attribute],hist=False,label=label1)
        sns.distplot(t1[attribute],hist=False,label=label2)
        plt.legend()
        plt.xlabel('Attribute',)
        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    plt.show()


# In[11]:


#corresponding to negative class
t0=train[train.target.values==0]
#corresponding to positive class
t1=train[train.target.values==1]
#train attributes from 2 to 102
train_attributes=train.columns.values[2:102]
#plot distribution of train attributes
plot_train_attribute_distribution(t0,t1,'0','1',train_attributes)


# In[12]:


#train attributes from 102 to 203
train_attributes=train.columns.values[102:203]
#plot distribution of train attributes
plot_train_attribute_distribution(t0,t1,'0','1',train_attributes)


# ###### Findings:
# * As you can see in the figure most variables either very closely, or somewhat imitate the normal distribution.
# *Variables like var_3, var_4, var_10, var_11,var_171,var_185 etc follow same distribution for both the classes of target
# *Variables like var_0, var_1,var_2, var_9, var_180, var_98 etc follow different distribution for both the classes of target

# ###### Predictor variables in test dataset 

# In[13]:


#Distribution of test attributes
def plot_test_attribute_distribution(test_attributes):
    i=0
    sns.set_style('whitegrid')
    
    fig=plt.figure()
    ax=plt.subplots(10,10,figsize=(22,18))
    
    for attribute in test_attributes:
        i+=1
        plt.subplot(10,10,i)
        sns.distplot(test[attribute],hist=False)
        plt.xlabel('Attribute',)
        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    plt.show()


# In[15]:


#test attribiutes from 1 to 101
test_attributes=test.columns.values[1:101]
#plot distribution of test attributes
plot_test_attribute_distribution(test_attributes)


# In[14]:


#test attributes from 101 to 202
test_attributes=test.columns.values[101:202]
#plot the distribution of test attributes
plot_test_attribute_distribution(test_attributes)


# ###### Let us look distribution of mean values per rows and columns in train and test dataset

# In[16]:


#Distribution of mean values per column in train and test dataset
plt.figure(figsize=(16,8))
#train attributes
train_attributes=train.columns.values[2:202]
#test attributes
test_attributes=test.columns.values[1:201]
#Distribution plot for mean values per column in train attributes
sns.distplot(train[train_attributes].mean(axis=0),color='green',kde=True,bins=150,label='train')
#Distribution plot for mean values per column in test attributes
sns.distplot(test[test_attributes].mean(axis=0),color='blue',kde=True,bins=150,label='test')
plt.title('Distribution of mean values per column in train and test dataset')
plt.legend()
plt.show()

#Distribution of mean values per row in train and test dataset
plt.figure(figsize=(16,8))
#Distribution plot for mean values per row in train attributes
sns.distplot(train[train_attributes].mean(axis=1),color='green',kde=True,bins=150,label='train')
#Distribution plot for mean values per row in test attributes
sns.distplot(test[test_attributes].mean(axis=1),color='blue',kde=True, bins=150, label='test')
plt.title('Distribution of mean values per row in train and test dataset')
plt.legend()
plt.show()


# ###### Let us look distribution of standard deviation(std) values per rows and columns in train and test dataset

# In[17]:


#Distribution of std values per column in train and test dataset
plt.figure(figsize=(16,8))
#train attributes
train_attributes=train.columns.values[2:202]
#test attributes
test_attributes=test.columns.values[1:201]
#Distribution plot for std values per column in train attributes
sns.distplot(train[train_attributes].std(axis=0),color='red',kde=True,bins=150,label='train')
#Distribution plot for std values per column in test attributes
sns.distplot(test[test_attributes].std(axis=0),color='blue',kde=True,bins=150,label='test')
plt.title('Distribution of std values per column in train and test dataset')
plt.legend()
plt.show()

#Distribution of std values per row in train and test dataset
plt.figure(figsize=(16,8))
#Distribution plot for std values per row in train attributes
sns.distplot(train[train_attributes].std(axis=1),color='red',kde=True,bins=150,label='train')
#Distribution plot for std values per row in test attributes
sns.distplot(test[test_attributes].std(axis=1),color='blue',kde=True, bins=150, label='test')
plt.title('Distribution of std values per row in train and test dataset')
plt.legend()
plt.show()


# ###### Let us look distribution of skewness per rows and columns in train and test dataset

# In[18]:


#Distribution of skew values per column in train and test dataset
plt.figure(figsize=(16,8))
#train attributes
train_attributes=train.columns.values[2:202]
#test attributes
test_attributes=test.columns.values[1:201]
#Distribution plot for skew values per column in train attributes
sns.distplot(train[train_attributes].skew(axis=0),color='green',kde=True,bins=150,label='train')
#Distribution plot for skew values per column in test attributes
sns.distplot(test[test_attributes].skew(axis=0),color='blue',kde=True,bins=150,label='test')
plt.title('Distribution of skewness values per column in train and test dataset')
plt.legend()
plt.show()

#Distribution of skew values per row in train and test dataset
plt.figure(figsize=(16,8))
#Distribution plot for skew values per row in train attributes
sns.distplot(train[train_attributes].skew(axis=1),color='green',kde=True,bins=150,label='train')
#Distribution plot for skew values per row in test attributes
sns.distplot(test[test_attributes].skew(axis=1),color='blue',kde=True, bins=150, label='test')
plt.title('Distribution of skewness values per row in train and test dataset')
plt.legend()
plt.show()


# ##### Let us look distribution of Kurtosis per rows and columns in train and test dataset

# In[19]:


#Distribution of kurtosis values per column in train and test dataset
plt.figure(figsize=(16,8))
#train attributes
train_attributes=train.columns.values[2:202]
#test attributes
test_attributes=test.columns.values[1:201]
#Distribution plot for kurtosis values per column in train attributes
sns.distplot(train[train_attributes].kurtosis(axis=0),color='blue',kde=True,bins=150,label='train')
#Distribution plot for kurtosis values per column in test attributes
sns.distplot(test[test_attributes].kurtosis(axis=0),color='green',kde=True,bins=150,label='test')
plt.title('Distribution of kurtosis values per column in train and test dataset')
plt.legend()
plt.show()

#Distribution of kutosis values per row in train and test dataset
plt.figure(figsize=(16,8))
#Distribution plot for kurtosis values per row in train attributes
sns.distplot(train[train_attributes].kurtosis(axis=1),color='blue',kde=True,bins=150,label='train')
#Distribution plot for kurtosis values per row in test attributes
sns.distplot(test[test_attributes].kurtosis(axis=1),color='green',kde=True, bins=150, label='test')
plt.title('Distribution of kurtosis values per row in train and test dataset')
plt.legend()
plt.show()


# ### Missing Value Analysis

# In[20]:


missingvalue_train=pd.DataFrame(train.isnull().sum())
missingvalue_test=pd.DataFrame(test.isnull().sum())

print (missingvalue_train)
print (missingvalue_test)


# As from above analysis, there are no missing values in train or test data

# ### Outlier Analysis

# In[21]:


def plot_feature_boxplot(df, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,5,figsize=(18,24))

    for feature in features:
        i += 1
        plt.subplot(10,5,i)
        sns.boxplot(df[feature]) 
        plt.xlabel(feature, fontsize=11)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', labelsize=6, pad=-6)
        plt.tick_params(axis='y', labelsize=6)
    plt.show()


# In[22]:


features = train.columns.values[2:52]
plot_feature_boxplot(train, features)
#From var_0 to var_49


# In[23]:


features = train.columns.values[52:102]
plot_feature_boxplot(train, features)
#From var_50 to var_99


# In[30]:


features = train.columns.values[102:152]
plot_feature_boxplot(train, features)
#From var_100 to var_149


# In[24]:


features = train.columns.values[152:202]
plot_feature_boxplot(train, features)
#From var_150 to var_199


# In[25]:


train_outliers=train


# In[26]:


Q1 = train_outliers.quantile(0.25)
Q3 = train_outliers.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[27]:


print("df.shape:",train_outliers.shape)
df_in = train_outliers[~((train_outliers < (Q1 - 1.5 * IQR)) |(train_outliers > (Q3 + 1.5 * IQR))).any(axis=1)]
df_out = train_outliers[((train_outliers < (Q1 - 1.5 * IQR)) |(train_outliers > (Q3 + 1.5 * IQR))).any(axis=1)]
print("df_in.shape:",df_in.shape)
print("df_out.shape:",df_out.shape)


# In[26]:


df_in['target'].value_counts()


# In[28]:


df_out['target'].value_counts()


# In[29]:


train_outliers['target'].value_counts()


# All the varaibles with target "1" are as considered outliers, In imbalanced class data, there is no requirement for outlier analysis

# ### Correlation Analysis

# ##### Correlation analysis for variables in train data

# In[31]:


train_corr=train.loc[:] 

f, ax = plt.subplots(figsize=(7, 5))
corr = train_corr.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# Findings: As from the heatmap it can be seen that the varaibles in train dataset are independent of each other

# ##### Correlation analysis for variables in test data

# In[32]:


test_corr=test.loc[:]

f, ax = plt.subplots(figsize=(7, 5))
corr = test_corr.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# Findings: As from the heatmap it can be seen that the varaibles in test dataset are independent of each other

# In[33]:


#Correlations in train attributes
train_attributes=train.columns.values[2:202]
train_correlations=train[train_attributes].corr().abs().unstack().sort_values(kind='quicksort').reset_index()
train_correlations=train_correlations[train_correlations['level_0']!=train_correlations['level_1']]
print(train_correlations.head(10))
print(train_correlations.tail(10))


# In[34]:


#Correlations in test attributes
test_attributes=test.columns.values[1:201]
test_correlations=test[test_attributes].corr().abs().unstack().sort_values(kind='quicksort').reset_index()
test_correlations=test_correlations[test_correlations['level_0']!=test_correlations['level_1']]
print(test_correlations.head(10))
print(test_correlations.tail(10))


# Findings: Clearly from the above correlation analysis we can see that the varaibles in train and test data have correlation values very less (<0.05)indicating they are independent of each other

# In[35]:


#Correlations in train data
train_correlations=train[train_attributes].corr()
train_correlations=train_correlations.values.flatten()
train_correlations=train_correlations[train_correlations!=1]
#Correlations in test data
test_correlations=test[test_attributes].corr()
test_correlations=test_correlations.values.flatten()
test_correlations=test_correlations[test_correlations!=1]

plt.figure(figsize=(20,5))
#Distribution plot for correlations in train data
sns.distplot(train_correlations, color="Red", label="train")
#Distribution plot for correlations in test data
sns.distplot(test_correlations, color="Blue", label="test")
plt.xlabel("Correlation values found in train and test")
plt.ylabel("Density")
plt.title("Correlation distribution plot for train and test attributes")
plt.legend()


# ### Feature Engineering

# ### Permutation importance

# Let's look at the permutation importance method of variable importance that changes the order of a column and measures the loss of accuracy of the model to estimate the importance of the feature set. Instead of dripping a feature like in a random forest method, the feature column in randomized. Intuitively, we should see same/similar set of features from both the techniques, ordered differently, and the performance shouldn't differ drastically. We'll use eli5 to compute importance.

# In[38]:


#training and testing data
X=train.drop(columns=['ID_code','target'],axis=1)
test=test.drop(columns=['ID_code'],axis=1)
y=train['target']


# In[39]:


#Split the training data
X_train,X_valid,y_train,y_valid=train_test_split(X,y,random_state=42)

print('Shape of X_train :',X_train.shape)
print('Shape of X_valid :',X_valid.shape)
print('Shape of y_train :',y_train.shape)
print('Shape of y_valid :',y_valid.shape)


# #### Random Forest Classifier

# In[40]:


#Random forest classifier
rf_model=RandomForestClassifier(n_estimators=10,random_state=42)
#fitting the model
rf_model.fit(X_train,y_train)


# In[41]:


#Permutation importance
from eli5.sklearn import PermutationImportance
perm_imp=PermutationImportance(rf_model,random_state=42)
#fitting the model
perm_imp.fit(X_valid,y_valid)


# In[44]:


#Important features
eli5.show_weights(perm_imp,feature_names=X_valid.columns.tolist(),top=200)


# Findings:
# * The variables in green rows have positive impact on our prediction
# * The variables in white rows have no impact on our prediction
# * The variables in red rows have negative impact on our prediction

# ### Handling of imbalanced data
Now we are going to explore 5 different approaches for dealing with imbalanced datasets.
1.Change the performance metric
2,Oversample minority class
3.Undersample majority class
4.Synthetic Minority Oversampling Technique(SMOTE)
5.Change the algorithm
6.Using StratifiedK-fold Cross ValidationFirst we will develop the Logistic regression model on imbalanced data and the model is evaluated by changing the evaluation metrics, then we will use SMOTE techniques to handle the imbalance data
# ## Logistic Regression using Stratified K-fold Cross Validation

# ###### Split the train data using StratefiedKFold cross validator

# In[45]:


#Training data
X=train.drop(['ID_code','target'],axis=1)
Y=train['target']
#StratifiedKFold cross validator
cv=StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
for train_index,valid_index in cv.split(X,Y):
    X_train, X_valid=X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid=Y.iloc[train_index], Y.iloc[valid_index]

print('Shape of X_train :',X_train.shape)
print('Shape of X_valid :',X_valid.shape)
print('Shape of y_train :',y_train.shape)
print('Shape of y_valid :',y_valid.shape)


# ###### Logistic Regression model

# In[46]:


#Logistic regression model
lr_model=LogisticRegression(random_state=42)
#fitting the lr model
lr_model.fit(X_train,y_train)


# In[47]:


#Accuracy of the model
lr_score=lr_model.score(X_train,y_train)
print('Accuracy of the lr_model :',lr_score)


# ###### Cross validation prediction of lr_model

# In[49]:


#Cross validation prediction
cv_predict=cross_val_predict(lr_model,X_valid,y_valid,cv=5)
#Cross validation score
cv_score=cross_val_score(lr_model,X_valid,y_valid,cv=5)
print('cross_val_score :',np.average(cv_score))


# ###### Confusion matrix

# In[50]:


#Confusion matrix
cm=confusion_matrix(y_valid,cv_predict)
#Plot the confusion matrix
plot_confusion_matrix(y_valid,cv_predict,normalize=False,figsize=(15,8))


# #### Reciever operating characteristics (ROC)-Area under curve(AUC) score and curve

# In[51]:


#ROC_AUC score
roc_score=roc_auc_score(y_valid,cv_predict)
print('ROC score :',roc_score)

#ROC_AUC curve
plt.figure()
false_positive_rate,recall,thresholds=roc_curve(y_valid,cv_predict)
roc_auc=auc(false_positive_rate,recall)
plt.title('Reciver Operating Characteristics(ROC)')
plt.plot(false_positive_rate,recall,'b',label='ROC(area=%0.3f)' %roc_auc)
plt.legend()
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall(True Positive Rate)')
plt.xlabel('False Positive Rate')
plt.show()
print('AUC:',roc_auc)


# When we compare the roc_auc_score and model accuracy , model is not performing well on imbalanced data.

# ###### Classification report

# In[52]:


#Classification report
scores=classification_report(y_valid,cv_predict)
print(scores)


# As from the table we can see that the f1 score is high for number of customers those who will not make a transaction then the who will make a transaction. So, we are going to change the algorithm.

# ###### Model performance on test data

# In[57]:


test=pd.read_csv("test.csv")


# In[58]:


#Predicting the model
X_test=test.drop(['ID_code'],axis=1)
lr_pred=lr_model.predict(X_test)
print(lr_pred)


# Oversample minority class:
# It can be defined as adding more copies of minority class.
# It can be a good choice when we don't have a ton of data to work with.
# Drawback is that we are adding information.This may leads to overfitting and poor performance on test data.
# 
# Undersample majority class:
# It can be defined as removing some observations of the majority class.
# It can be a good choice when we have a ton of data -think million of rows.
# Drawback is that we are removing information that may be valuable.This may leads to underfitting and poor performance on test data.
# 
# Both Oversampling and undersampling techniques have some drawbacks. So, we are not going to use this models for this problem and also we will use other best algorithms.

# ### Synthetic Minority Oversampling Technique(SMOTE)

# SMOTE uses a nearest neighbors algorithm to generate new and synthetic data to used for training the model.We'll use SMOTE method to generate some data points in the minority class and see if the model improves

# In[59]:


#Synthetic Minority Oversampling Technique
sm = SMOTE(random_state=42, ratio=1.0)
#Generating synthetic data points
X_smote,y_smote=sm.fit_sample(X_train,y_train)
X_smote_v,y_smote_v=sm.fit_sample(X_valid,y_valid)


# Let us see how baseline logistic regression model performs on synthetic data points

# In[60]:


#Logistic regression model for SMOTE
smote=LogisticRegression(random_state=42)
#fitting the smote model
smote.fit(X_smote,y_smote)


# In[61]:


#Accuracy of the model
smote_score=smote.score(X_smote,y_smote)
print('Accuracy of the smote_model :',smote_score)


# Cross validation prediction of smoth_model

# In[62]:


#Cross validation prediction
cv_pred=cross_val_predict(smote,X_smote_v,y_smote_v,cv=5)
#Cross validation score
cv_score=cross_val_score(smote,X_smote_v,y_smote_v,cv=5)
print('cross_val_score :',np.average(cv_score))


# ###### Confusion matrix

# In[63]:


#Confusion matrix
cm=confusion_matrix(y_smote_v,cv_pred)
#Plot the confusion matrix
plot_confusion_matrix(y_smote_v,cv_pred,normalize=False,figsize=(15,8))


# #### Reciever operating characteristics (ROC)-Area under curve(AUC) score and curve

# In[64]:


#ROC_AUC score
roc_score=roc_auc_score(y_smote_v,cv_pred)
print('ROC score :',roc_score)

#ROC_AUC curve
plt.figure()
false_positive_rate,recall,thresholds=roc_curve(y_smote_v,cv_pred)
roc_auc=auc(false_positive_rate,recall)
plt.title('Reciver Operating Characteristics(ROC)')
plt.plot(false_positive_rate,recall,'b',label='ROC(area=%0.3f)' %roc_auc)
plt.legend()
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall(True Positive Rate)')
plt.xlabel('False Positive Rate')
plt.show()
print('AUC:',roc_auc)


# #### Classification report

# In[65]:


#Classification report
scores=classification_report(y_smote_v,cv_pred)
print(scores)


# #### Model performance on test data

# In[67]:


#Predicting the model
X_test=test.drop(['ID_code'],axis=1)
smote_pred=smote.predict(X_test)
print(smote_pred)


# SMOTE model is performing better than the logistic regression

# ###### It can be seen that SMOTE model is performing better than Logistic regression

# ### LightGBM:

# LightGBM is a gradient boosting framework that uses tree based learning algorithms. We are going to use LightGBM model.
# 
# Let us build LightGBM model

# In[68]:


#Training the model
#training data
lgb_train=lgb.Dataset(X_train,label=y_train)
#validation data
lgb_valid=lgb.Dataset(X_valid,label=y_valid)


# ###### choosing of hyperparameters

# In[69]:


#Selecting best hyperparameters by tuning of different parameters
params={'boosting_type': 'gbdt', 
          'max_depth' : -1, #no limit for max_depth if <0
          'objective': 'binary',
          'boost_from_average':False, 
          'nthread': 20,
          'metric':'auc',
          'num_leaves': 50,
          'learning_rate': 0.01,
          'max_bin': 100,      #default 255
          'subsample_for_bin': 100,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'bagging_fraction':0.5,
          'bagging_freq':5,
          'feature_fraction':0.08,
          'min_split_gain': 0.45, #>0
          'min_child_weight': 1,
          'min_child_samples': 5,
          'is_unbalance':True,
          }


# In[70]:


num_rounds=10000
lgbm= lgb.train(params,lgb_train,num_rounds,valid_sets=[lgb_train,lgb_valid],verbose_eval=1000,early_stopping_rounds = 5000)
lgbm


# ###### Training the lgbm model

# #### lgbm model performance on test data

# In[71]:


#predict the model
X_test=test.drop(['ID_code'],axis=1)
#probability predictions
lgbm_predict_prob=lgbm.predict(X_test,random_state=42,num_iteration=lgbm.best_iteration)
#Convert to binary output 1 or 0
lgbm_predict=np.where(lgbm_predict_prob>=0.5,1,0)
print(lgbm_predict_prob)
print(lgbm_predict)


# Conclusion :
# We tried model with logistic regression,smote and lightgbm. But lightgbm model is performing well on imbalanced data compared to other models based on scores of roc_auc_score.

# ###### Let us plot the important features

# In[72]:


#plot the important features
lgb.plot_importance(lgbm,max_num_features=50,importance_type="split",figsize=(20,50))


# In[73]:


#final submission
submission_df=pd.DataFrame({'ID_code':test['ID_code'].values})
submission_df['lgbm_predict_prob']=lgbm_predict_prob
submission_df['lgbm_predict']=lgbm_predict
submission_df.to_csv('submission.csv',index=False)
submission_df.head()


# ### Model explaining

# ##### Partial dependency of features
Partial dependence plot gives a graphical depiction of the marginal effect of a variable on the class probability or classification.While feature importance shows what variables most affect predictions, but partial dependence plots show how a feature affects predictions.
Let us calculate partial dependence plots on random forest
# In[74]:


#Create the data we will plot 'var_53'
features=[v for v in X_valid.columns if v not in ['ID_code','target']]
pdp_data=pdp.pdp_isolate(rf_model,dataset=X_valid,model_features=features,feature='var_53')
#plot feature "var_53"
pdp.pdp_plot(pdp_data,'var_53')
plt.show()


# In[75]:


#Create the data we will plot 
pdp_data=pdp.pdp_isolate(rf_model,dataset=X_valid,model_features=features,feature='var_6')
#plot feature "var_6"
pdp.pdp_plot(pdp_data,'var_6')
plt.show()

The y_axis does not show the predictor value instead how the value changing with the change in given predictor variable. 
The blue shaded area indicates the level of confidence of 'var_53', 'var_6'
On y-axis having a positive value means for that particular value of predictor variable it is less likely to predict the correct class and having a positive value means it has positive impact on predicting the correct class.
# In[ ]:




