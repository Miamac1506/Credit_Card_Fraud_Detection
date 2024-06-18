#!/usr/bin/env python
# coding: utf-8

# In[67]:


#Importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score


# In[3]:


#Loading the dataset to a Pandas DataFrame
credit_df = pd.read_csv('/Users/macminhanh/Downloads/creditcard.csv')

print(credit_df.head())


# In[4]:


#Getting information about the dataset
credit_df.info()


# In[5]:


#Checking if there are any missing values in each collum
credit_df.isnull().sum()


# In[6]:


#Checking the distribution of legit transactions and fraud transactions
credit_df['Class'].value_counts()


# In[7]:


#As we can see, the dataset is unbalanced, which makes it hard to apply machine learning. Applying machine learning models directly to such a skewed dataset will likely result in a model biased towards predicting transactions as legitimate, simply because they are the overwhelming majority. This could lead to high accuracy in predicting legitimate transactions while failing to adequately identify fraudulent ones, which are typically the focus in fraud detection.


# In[ ]:


#DATA ANALYSIS


# In[8]:


# Separting the data for analysis
legit = credit_df[credit_df.Class == 0 ]
fraud = credit_df[credit_df.Class == 1 ]


# In[9]:


print(legit.shape)
print(fraud.shape)


# In[13]:


#Statistical measures of the data
legit.Amount.describe()


# In[14]:


fraud.Amount.describe()


# In[15]:


#Comparing the values for both transactions
credit_df.groupby('Class').mean()


# In[16]:


# These features show significant differences in their means between the two classes. For example, V1 has a mean of -4.77 for fraudulent transactions, compared to 0.01 for legitimate ones. This stark contrast highlights that these components capture crucial differences in transaction patterns between frauds and legitimate transactions.
#The differences in averages across many of the features for the two classes of transactions underscore the importance of these features in distinguishing between legitimate and fraudulent activities. Machine learning models can leverage these differences to predict whether a new transaction is likely to be fraudulent.


# In[17]:


#DEALING WITH THE UNBALANCED DATASET (under-sampling technique)
#Building a sample dataset from the original dataset, containing similar distribution of normal transactions and fraudulent transaction.

legit_sample = legit.sample(n=492)


# In[18]:


#Concatenting two DataFrames
new_df = pd.concat([legit_sample, fraud], axis=0)


# In[19]:


new_df.head()


# In[20]:


# Now the new DataFrame that has an equal number of legitimate and fraudulent transactions. Each class now has 492 instances, making the dataset balanced


# In[21]:


new_df['Class'].value_counts()


# In[22]:


new_df.groupby('Class').mean()


# In[23]:


# With the new dataset, the mean values are only slightly different, indicating that this new dataset reflects the original dataset pretty well and can be used for machine learning.


# In[24]:


#Splitting the new dataset into features and targets

X = new_df.drop(columns='Class', axis=1) #features
Y = new_df['Class'] #targets

print(X)


# In[25]:


print(Y)


# In[27]:


#Splitting the data into training data and testing data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)


# In[28]:


print(X.shape, X_train.shape, X_test.shape)


# In[29]:


#Model Training
#Logistic Regression

model = LogisticRegression()


# In[30]:


#Training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# In[33]:


#Model Evalutaion based on Accuracy Score
#Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training Data:',training_data_accuracy)


# In[34]:


#Accuracy on testing data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Testing Data:', testing_data_accuracy)


# In[55]:


#Model Evaluation based on Precision Score - what proportion of transactions that the model predicted as fraudulent were actually fraudulent.


#Precision on training data
lr_train_precision = precision_score(Y_train, X_train_prediction)
print('Logistic Regression Training Precision:', lr_train_precision)


# In[57]:


#Precision on testing data
lr_test_precision = precision_score(Y_test, X_test_prediction)
print('Logistic Regression Testing Precision:', lr_test_precision)


# In[58]:


#Model Evaluation based on Recall Score  - the ability of the model to correctly identify all actual cases of fraud

#Recall Score on training data
lr_train_recall = recall_score(Y_train, X_train_prediction)
print('Logistic Regression Training Recall:', lr_train_recall)


# In[59]:


#Recall Score on testing data
lr_test_recall = recall_score(Y_test, X_test_prediction)
print('Logistic Regression Testing Recall:', lr_test_recall)


# In[60]:


#Model Evaluation based on F1 Score - overall performance of the machine learning model by balancing precision and recall

#F1 Score on training data
lr_train_f1 = f1_score(Y_train, X_train_prediction)
print('Logistic Regression Training F1 Score:', lr_train_f1)


# In[61]:


#F1 Score on testing data
lr_test_f1 = f1_score(Y_test, X_test_prediction)
print('Logistic Regression Testing F1 Score:', lr_test_f1)


# In[37]:


#Decision Trees
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=2)


# In[38]:


# Training the Decision Tree model
dt_model.fit(X_train, Y_train)


# In[41]:


#Accuracy on training data
dt_train_predictions = dt_model.predict(X_train)
dt_train_accuracy = accuracy_score(Y_train, dt_train_predictions)
print('Decision Tree Accuracy on Training Data:', dt_train_accuracy)


# In[ ]:


#This result indicates that the Decision Tree model has achieved perfect accuracy on the training dataset, meaning it correctly classified every single instance in the training data.
#While a training accuracy of 1.0 might initially seem excellent, it is often indicative of overfitting


# In[45]:


#Accuracy on testing data
dt_test_predictions = dt_model.predict(X_test)
dt_test_accuracy = accuracy_score(Y_test, dt_test_predictions)
print('Decision Tree Accuracy on Testing Data:', dt_test_accuracy)


# In[ ]:


#The accuracy on the testing data is considerably lower than on the training data. An accuracy of about 88.83% is still quite high, but the discrepancy suggests that the model does not perform as well on unseen data, reinforcing the suspicion of overfitting.


# In[64]:


#Precision Score on traing and testing data

dt_train_precision = precision_score(Y_train, dt_train_predictions)
dt_test_precision = precision_score(Y_test, dt_test_predictions)

print('Decision Tree Training Precision:', dt_train_precision)
print('Decision Tree Testing Precision:', dt_test_precision)


# In[65]:


# Recall Score on training and testing data

dt_train_recall = recall_score(Y_train, dt_train_predictions)
dt_test_recall = recall_score(Y_test, dt_test_predictions)

print('Decision Tree Training Recall:', dt_train_recall)
print('Decision Tree Testing Recall:', dt_test_recall)


# In[66]:


# F1 Score on training and testing data

dt_train_f1 = f1_score(Y_train, dt_train_predictions)
dt_test_f1 = f1_score(Y_test, dt_test_predictions)

print('Decision Tree Training F1 Score:', dt_train_f1)
print('Decision Tree Testing F1 Score:', dt_test_f1)


# In[62]:


#XGBoost

get_ipython().system('pip install xgboost')
import xgboost as xgb
xgb_model = xgb.XGBClassifier(random_state=2, use_label_encoder=False, eval_metric='logloss')


# In[48]:


# Training the XGBoost model
xgb_model.fit(X_train, Y_train)


# In[52]:


#Accuracy on training data
xgb_train_predictions = xgb_model.predict(X_train)
xgb_train_accuracy = accuracy_score(Y_train, xgb_train_predictions)
print('XGBoost Accuracy on Training Data:', xgb_train_accuracy)


# In[53]:


#Accuracy on testing data
xgb_test_predictions = xgb_model.predict(X_test)
xgb_test_accuracy = accuracy_score(Y_test, xgb_test_predictions)
print('XGBoost Accuracy on Testing Data:', xgb_test_accuracy)


# In[ ]:


#The accuracy on the testing data is about 91.88%, which is quite high but noticeably lower than the training accuracy. This disparity typically confirms the suspicion of overfitting but to a lesser extent than what might be seen with simpler models like decision trees without ensemble methods.


# In[69]:


# Precision Score on testing and training data
xgb_train_precision = precision_score(Y_train, xgb_train_predictions)
xgb_test_precision = precision_score(Y_test, xgb_test_predictions)

print('XGBoost Training Precision:', xgb_train_precision)
print('XGBoost Testing Precision:', xgb_test_precision)


# In[70]:


# Recall Score on testing and training data
xgb_train_recall = recall_score(Y_train, xgb_train_predictions)
xgb_test_recall = recall_score(Y_test, xgb_test_predictions)

print('XGBoost Training Recall:', xgb_train_recall)
print('XGBoost Testing Recall:', xgb_test_recall)


# In[71]:


# F1 Score on testing and training data
xgb_train_f1 = f1_score(Y_train, xgb_train_predictions)
xgb_test_f1 = f1_score(Y_test, xgb_test_predictions)

print('XGBoost Training F1 Score:', xgb_train_f1)
print('XGBoost Testing F1 Score:', xgb_test_f1)


# In[ ]:




