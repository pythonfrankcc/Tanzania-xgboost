
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing the necessary dependencies
import xgboost as xgb#the algorithm to be used
import matplotlib.pyplot as plt#for visualization


# In[ ]:


#let's now read the data
#loading the train data
train_data = pd.read_csv('../input/training.csv')
print("The train data")
train_data.head()


# In[ ]:


#loading the test data
test_data = pd.read_csv('../input/test (1).csv')
print("The test data")
test_data.head()


# In[ ]:


#information on the train_features
train_data.info()


# In[ ]:


#information on the test_features
test_data.info()


# In[ ]:


#from the above we can see that train_Data has more columns
# check if the data has missing points with seaborn heatmap
import seaborn as sns
sns.heatmap(train_data.isnull(),yticklabels=False, cmap='viridis')


# In[ ]:


# view the columns in train_data
train_data.columns


# In[ ]:


# view the columns in test_data
test_data.columns


# In[ ]:


#lets check how thye submission csv is supposed to look like 
submission= pd.read_csv('../input/sample_submission (2).csv')
submission.head()


# In[ ]:


#dropping the labels that you are supposed to predict and the excess from train_head
cols = ['ID','mobile_money', 'savings', 'borrowing','insurance']
train_data = train_data.drop(cols, axis=1)
x_test = test_data.drop(['ID'], axis=1)


# In[ ]:


train_data.head()


# In[ ]:


X = train_data.drop(['mobile_money_classification'], axis=1)
y = train_data['mobile_money_classification']


# In[ ]:


y.head()


# In[ ]:


'''#check for the necessary and important features and look for the ones to discard
import xgboost as xgb
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=1000,booster='gbtree',max_depth=5,subsample=1,
    objective= 'multi:softmax',num_class=4,n_gpus= 0)
# error evaluation for multiclass training
xgb.fit(X, y)
feature_importances = pd.DataFrame(xgb.feature_importances_,
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)'''


# In[ ]:


#lets drop the most irrelevant columns in both X and the x_test
#from the already done tests we find that dropping the last three is what works best
X = X.drop(['Q8_11','Q8_7','Q8_5'], axis=1)
x_test = x_test.drop(['Q8_11','Q8_7','Q8_5'], axis=1)


# In[ ]:


#making sure that you have dropped the columns
X.columns,x_test.columns


# In[ ]:


#lets normalize the datasets
from sklearn.preprocessing import MinMaxScaler
names=X.columns
names1=x_test.columns
scaler = MinMaxScaler(feature_range=(0, 1))
X1 = scaler.fit_transform(X)
X2 = pd.DataFrame(X1, columns=names)
X_test=scaler.fit_transform(x_test)
X_test1 = pd.DataFrame(X_test, columns=names1)


# In[ ]:


y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y, random_state=42, stratify=y)

X_train.shape, y_train.shape, X_test.shape, y_test.shape,


# In[ ]:


import xgboost as xgb

from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=1000,max_depth= 4,
    objective= 'multi:softmax',
    sub_sample=1,                
    num_class= 4,
    n_gpus= 0)
# error evaluation for multiclass training
xgb.fit(X_train,y_train)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_lables_trf = labels.transform(y_train)

test_pred = pd.DataFrame(xgb.predict_proba(X_test),columns=labels.classes_)


# In[ ]:


#test_pred = pd.DataFrame(bst.predict(X_test1), columns=labels.classes_)
q = {'ID': test_data["ID"], 'no_financial_services': test_pred[0], 'other_only': test_pred[1],
    'mm_only': test_pred[2], 'mm_plus': test_pred[3]}
df_pred1 = pd.DataFrame(data=q)
df_pred1 = df_pred1[['ID','no_financial_services', 'other_only', 'mm_only', 'mm_plus'  ]]


# In[ ]:


df_pred1.head()


# In[ ]:


df_pred1.to_csv('pred_set.csv', index=False) #save to csv file#

