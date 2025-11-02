#!/usr/bin/env python
# coding: utf-8

# # Churn Model Training
# - This notebook trains a logistic regression model for churn prediction using K-Fold cross-validation with hard coded C(regularization strength)

# ## Import Needed Libraries
# - pandas
# - numpy
# - sklearn
# 
# Dataset Link: https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/chapter-03-churn-prediction

# In[1]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split               # --> data splitting
from sklearn.model_selection import KFold                           # --> create folds

from sklearn.feature_extraction import DictVectorizer             # --> handle categorical variables
from sklearn.linear_model import LogisticRegression               # --> logistic model
from sklearn.metrics import roc_auc_score                         # --> evaluate with auc_roc_score            


# In[2]:


df = pd.read_csv('../03-churn-project/WA_Fn-UseC_-Telco-Customer-Churn.csv', na_values=['',' '])

df.columns  = df.columns.str.lower() #convert columns to lower case

categorical_columns = list(df.dtypes[df.dtypes == "object"].index) # get all categorical columns

# For each column name, Replace empty spaces in column indices with _
for cols in categorical_columns:
    df[cols] = df[cols].str.lower().str.replace(' ','_') 

# replace all missing entries in totalcharges column with the median   
df['totalcharges'] = df.totalcharges.fillna(df.totalcharges.median())

#convert the entries in churn column into integer (0,1)
df.churn = (df.churn == 'yes').astype(int)


# In[3]:


df_full_train, df_test = train_test_split(df, test_size=0.2,random_state=1)

#y_test = df_test.churn.values

# df_full_train.shape, df_test.shape

# df_f_train, df_tes = train_test_split(df, train_size=0.8, random_state=1)

# df_f_train.shape, df_tes.shape


# In[4]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
     'gender',
     'seniorcitizen',
     'partner',
     'dependents',
     'phoneservice',
     'multiplelines',
     'internetservice',
     'onlinesecurity',
     'onlinebackup',
     'deviceprotection',
     'techsupport',
     'streamingtv',
     'streamingmovies',
     'contract',
     'paperlessbilling',
     'paymentmethod',
]


# In[5]:


# training pipeline - function that accepts X, y C,
# get the dictvectorizer on train, fit the logistic model, returns dv, model

def train(df_train, y_train, C=0.5):
    dicts = df_train[numerical + categorical].to_dict(orient = 'records')

    dv = DictVectorizer(sparse = False)

    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=0.5, max_iter=10000).fit(X_train, y_train)

    return dv, model


# In[6]:


#prediction function

def predict(df_val, dv, model):
    dicts = df_val[numerical + categorical].to_dict(orient = 'records')
    X_val = dv.transform(dicts)

    y_pred = model.predict_proba(X_val)[:,1]

    return y_pred


# In[7]:


C = 0.5
n_splits = 5


# In[8]:


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C) #--> the train function we wrote
    y_pred = predict(df_val, dv, model)       #--> the predict function we wrote

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C = %s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[9]:


scores


# In[10]:


dv, model = train(df_full_train, df_full_train.churn.values, C=0.5)
y_pred = predict(df_test, dv, model)
y_test = df_test.churn.values

auc = roc_auc_score(y_test, y_pred); auc


# In[ ]:





# In[12]:





# ## Save the model

# In[11]:


import pickle
output_file = f'model_C={C}.bin'
output_file


# In[23]:


# f_out = open(output_file, 'wb')  #open file and write its contents(dv & model) as binaries 
# pickle.dump((dv, model), f_out)
# f_out.close()


# In[13]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
    #do stuff

#do other stuff


# In[27]:


dv, model


# # Load the model

# In[1]:


import pickle

model_file = 'model_C=0.5.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[2]:


dv, model


# In[3]:


customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbiling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# In[4]:


X = dv.transform([customer])


# In[5]:


model.predict_proba(X)[0,1]


# In[ ]:




