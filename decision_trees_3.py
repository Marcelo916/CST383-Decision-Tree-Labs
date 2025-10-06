# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 15:32:19 2025

CST 383: Lab - Decision Trees Part 3

@author: Marcelo Villalobos Diaz
"""


"""
INSTRUCTIONS:
    
- pick some predictors, and create training and test sets
- create a DecisionTreeRegressor, and set some of the hyperparameters listed in lecture:
- min_samples_split, max_depth, min_samples_leaf, max_leaf_nodes, min_impurity_decrease
- train your tree and see what it looks like
- compute the RMSE for your model on the test data  ("test RMSE")
"""


import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import graphviz

# Seaborn plot settings
sns.set()
sns.set_context('talk')
rcParams['figure.figsize'] = 10, 8

# Load CPU data
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor'] + ' ' + df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3 / df['myct'], 2)

# Explore a few predictors
predictors = ['mmin', 'chmax', 'cs']
target = 'prp'

X = df[predictors].values
y = df[target].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train DecisionTreeRegressor
reg = DecisionTreeRegressor(
    max_depth=4, 
    min_samples_split=5,
    min_samples_leaf=3,
    max_leaf_nodes=10,
    random_state=0
)
reg.fit(X_train, y_train)

dot_data = export_graphviz(
    reg, 
    precision=2,
    feature_names=predictors,
    proportion=True,
    filled=True, 
    rounded=True
)
graph = graphviz.Source(dot_data)
graph.render("cpu_tree", format="png", cleanup=True)
graph

# Predict and compute RMSE
y_predict = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print(f"Test RMSE: {rmse:.2f}")

# Try a few more hyperparameter combinations

# Small tree
reg_small = DecisionTreeRegressor(max_depth=2, random_state=0)
reg_small.fit(X_train, y_train)
y_predict_small = reg_small.predict(X_test)
rmse_small = np.sqrt(mean_squared_error(y_test, y_predict_small))
print(f"Small tree RMSE: {rmse_small:.2f}")

# Large tree
reg_large = DecisionTreeRegressor(max_depth=10, min_samples_split=2, random_state=0)
reg_large.fit(X_train, y_train)
y_predict_large = reg_large.predict(X_test)
rmse_large = np.sqrt(mean_squared_error(y_test, y_predict_large))
print(f"Large tree RMSE: {rmse_large:.2f}")

