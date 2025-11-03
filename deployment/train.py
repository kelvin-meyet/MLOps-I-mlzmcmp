#!/usr/bin/env python
# coding: utf-8


# # Churn Model Training
# - This notebook trains a logistic regression model for churn prediction using K-Fold cross-validation with hard coded C(regularization strength)

# Import Needed Libraries
# - pandas
# - numpy
# - sklearn

# Dataset Link-> https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/chapter-03-churn-prediction

print("importing libraries.....")


import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split  # --> data splitting
from sklearn.model_selection import KFold  # --> create folds

from sklearn.feature_extraction import (
    DictVectorizer,
)  # --> handle categorical variables
from sklearn.linear_model import LogisticRegression  # --> logistic model
from sklearn.metrics import roc_auc_score  # --> evaluate with auc_roc_score


print("")
print("reading hardcoded parameters......")
# --> Parameter
C = 0.5
n_splits = 5
output_file = f"model_C={C}.bin"


print("")
print("Data loading and data preparation begins.....")

# --- Data Preparation----
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", na_values=["", " "])

df.columns = df.columns.str.lower()
categorical_columns = list(df.dtypes[df.dtypes == "object"].index)

# For each column name, Replace empty spaces in column indices with _
for cols in categorical_columns:
    df[cols] = df[cols].str.lower().str.replace(" ", "_")

# replace all missing entries in totalcharges column with the median
df["totalcharges"] = df.totalcharges.fillna(df.totalcharges.median())

# convert the entries in churn column into integer (0,1)
df.churn = (df.churn == "yes").astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

numerical = ["tenure", "monthlycharges", "totalcharges"]

categorical = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
]

print("data preping ends....")
print("")


print("preparing training and prediction functions for pipeline.... ")

# ----> training pipeline - function that accepts X, y C, ---
# --- > get the dictvectorizer on train, fit the logistic model, returns dv, model


def train(df_train, y_train, C=0.5):
    dicts = df_train[numerical + categorical].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=0.5, max_iter=10000).fit(X_train, y_train)

    return dv, model


# ---> prediction function
def predict(df_val, dv, model):
    dicts = df_val[numerical + categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)

    y_pred = model.predict_proba(X_val)[:, 1]

    return y_pred


print("")
print(f"cross validation starts with hardcoded C={C}")
# Cross validation validation pipeline

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)  # --> the train function we wrote
    y_pred = predict(df_val, dv, model)  # --> the predict function we wrote

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f"auc on fold {fold} is {auc}")
    # print(f'auc on fold {fold} is {auc}')
    fold = fold + 1


print("validation results")
print("C = %s %.3f +- %.3f" % (C, np.mean(scores), np.std(scores)))


print("Training the final model on the full training data")
# Training the final model on the full training data
dv, model = train(df_full_train, df_full_train.churn.values, C=0.5)


print("predict on held-out test data")
# Predict on held out test data
y_pred = predict(df_test, dv, model)
y_test = df_test.churn.values

auc = roc_auc_score(y_test, y_pred)

print(f"auc on test data is = {auc}")


# ---- Save the model -----
with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)

print(f"the model is saved to {output_file}")

# we will load this model(DictVectorizer & model) for predicting services
