#!/usr/bin/env python
# coding: utf-8

# # Churn Model Building
print("Loading Libraries ...")
import pandas as pd
import numpy as np
import sklearn

print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


from sklearn.model_selection import train_test_split               # --> data splitting
from sklearn.model_selection import KFold                           # --> create folds

from sklearn.feature_extraction import DictVectorizer             # --> handle categorical variables
from sklearn.linear_model import LogisticRegression               # --> logistic model
from sklearn.metrics import roc_auc_score                         # --> evaluate with auc_roc_score    
from sklearn.pipeline import make_pipeline

#--> Get Data
print("Reading data ...")

#get_ipython().system("wget 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/refs/heads/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'")


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv', na_values=['', ' '])

df.columns = df.columns.str.lower()

string_cols = list(df.dtypes[df.dtypes=='object'].index)

for col in string_cols:
    df[col] = df[col].str.lower().str.replace(' ', '_')


#--> Clean Data
print('Cleaning data ...')

#Replace missingness with in total charges with median value
df['totalcharges'] = df['totalcharges'].fillna(df['totalcharges'].median())
#convert target variable from string (yes/no)to integer(0/1)
df.churn = (df.churn == 'yes').astype(int)
#check missingness
#print(df.isnull().sum())


#--> Data Partition - 80:20 -> train & eval : test 
print("Partitioning data ...")
df_full_train, df_test = train_test_split(df, test_size=0.2,random_state=1)
df_full_train.shape, df_test.shape


#--> specify categorical and numerical features
print("Specify categorical and numerical variables...")

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


# --> Functions needed for CV
print("Unwrapping functions for Cross validation ...")
# training pipeline - function that accepts X, y, C,
# get the dictvectorizer on train, fit the logistic model, returns dv, model

def train(df_train, y_train, C=0.5):
    train_dicts = df_train[numerical + categorical].to_dict(orient = 'records')

    dv = DictVectorizer(sparse = False)

    X_train = dv.fit_transform(train_dicts)

    model = LogisticRegression(C=0.5, max_iter=10000).fit(X_train, y_train)

    return dv, model


#prediction function
def predict(df_val, dv, model):
    val_dicts = df_val[numerical + categorical].to_dict(orient = 'records')
    X_val = dv.transform(val_dicts)

    y_pred = model.predict_proba(X_val)[:,1]

    return y_pred


print("CV begins ... " )
C = 0.5
n_splits = 5

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
    
print("best parameters are: ")
print('C = %s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# ---> Fit a final model with best parameters from CV
print("use best parameters from CV for pipelines ...")
# # Option 2 - Efficient Pipeline 
# ## Fit a final model with pipelines from sklearn - feed the parameters from the CV done prior

best_C = C
training_pipeline = make_pipeline(
    DictVectorizer(sparse = False),
    LogisticRegression(C=best_C, max_iter=10000)    #solver='liblinear'
)

print(training_pipeline)

y_train = df_full_train.churn.values
train_dict = df_full_train[categorical + numerical].to_dict(orient='records')
training_pipeline.fit(train_dict, y_train)


#--> Test the model on held out test set
print('Testing the model on heldout test set ...')

test_dict = df_test[categorical + numerical].to_dict(orient = 'records')
y_pred_pipe = training_pipeline.predict_proba(test_dict)[:,1]
y_test = df_test.churn.values

auc = roc_auc_score(y_test, y_pred_pipe)
print('auc on held out test data is', auc)


#--> Save pipeline model
import pickle
output_file = f'pipeline_model_C={C}.bin'
#print(output_file)
with open(output_file, 'wb') as f_out:
    pickle.dump((training_pipeline), f_out)
    
print(f'Saved the model as {output_file} ...')



