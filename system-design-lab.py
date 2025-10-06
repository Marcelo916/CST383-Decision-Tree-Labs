# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:10:24 2019

CST 383: Lab - System Design

@author: Marcelo Villalobos Diaz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# Steps 1 - 4

# read the data
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/german-credit.csv")
bad_loan = df['good.loan'] - 1

# use only numeric data, and scale it
df = df[["duration.in.months", "amount", "percentage.of.disposable.income", "at.residence.since", 
              "age.in.years", "num.credits.at.bank"]]
X = df.apply(zscore).values
y = bad_loan.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# see how knn classifier works as training size changes


ks = [1, 3, 5, 9]

plt.figure(figsize=(12,10))

for i, k in enumerate(ks):
    knn = KNeighborsClassifier(n_neighbors=k)
    te_errs = []
    tr_errs = []
    tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)
    for tr_size in tr_sizes:
        X_train1 = X_train[:tr_size,:]
        y_train1 = y_train[:tr_size]
        knn.fit(X_train1, y_train1)
        tr_predicted = knn.predict(X_train1)
        tr_err = (tr_predicted != y_train1).mean()
        tr_errs.append(tr_err)
        te_predicted = knn.predict(X_test)
        te_err = (te_predicted != y_test).mean()
        te_errs.append(te_err)
    
    plt.subplot(2, 2, i+1)
    plt.plot(tr_sizes, tr_errs, label='Train Error')
    plt.plot(tr_sizes, te_errs, label='Test Error')
    plt.title(f'Learning Curve (k={k})')
    plt.xlabel("Training Size")
    plt.ylabel("Error Rate")
    plt.legend()

plt.tight_layout()
plt.show()
  

# 5. Explain the curves you get.  For example, what’s with the low training error when k = 1?
"""
When k=1, the model memorizes the training data, so training error is close to zero. But test 
error is much higher because the model overfits. As k increases, both train and test error get 
closer together.
"""

# 6. What do the learning curves tell you?   Write some sentences to explain what the learning 
# curves tell you about bias and variance.
"""
The learning curves show how model performance changes with more training data. If the gap 
between training and test error is large, the model has high variance. If both errors are high, 
the model has high bias.
"""

# 7. Add some more features, and produce the 4 learning curve plots again.  
# original feature set (unchanged original dataset)
# Load full dataset again
df_full = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/german-credit.csv")

df = df_full[["duration.in.months", "amount", "percentage.of.disposable.income", 
              "at.residence.since", "age.in.years", "num.credits.at.bank"]]

df_updated = df.copy()
df_updated["checking.status"] = pd.factorize(df_full["checking.status"])[0]

bad_loan = df_full['good.loan'] - 1

X = df_updated.apply(zscore).values
y = bad_loan.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

ks = [1, 3, 5, 9]

plt.figure(figsize=(12,10))

for i, k in enumerate(ks):
    knn = KNeighborsClassifier(n_neighbors=k)
    te_errs = []
    tr_errs = []
    tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)
    
    for tr_size in tr_sizes:
        X_train1 = X_train[:tr_size,:]
        y_train1 = y_train[:tr_size]
        knn.fit(X_train1, y_train1)
        tr_predicted = knn.predict(X_train1)
        tr_err = (tr_predicted != y_train1).mean()
        tr_errs.append(tr_err)
        te_predicted = knn.predict(X_test)
        te_err = (te_predicted != y_test).mean()
        te_errs.append(te_err)
    
    plt.subplot(2, 2, i+1)
    plt.plot(tr_sizes, tr_errs, label='Train Error')
    plt.plot(tr_sizes, te_errs, label='Test Error')
    plt.title("Learning Curve (k={k})")
    plt.xlabel("Training Size")
    plt.ylabel("Error Rate")
    plt.legend()

plt.tight_layout()
plt.show()

