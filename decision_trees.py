# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 10:10:15 2025

CST 383: Lab - Decision Trees 

@author: Marcelo Villalobos Diaz
"""


# 1. Explain why decision trees are non-parametric models.
"""
Decision trees are nonparametric because they do not assume any specific form for 
the data distribution. They learn the structure directly from the data by splitting 
on feature values, so they can model complex relationships without requiring parameters 
that define a fixed equation.
"""


# 2. Use this code to read and preprocess the data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import graphviz

df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/College.csv', index_col=0)


# 3. Convert the 'Private' column to an numeric column with values 0 and 1 (1 for private colleges).
df['Private'] = (df['Private'] == 'Yes').astype(int)


# 4. Do a little exploration of the data to remember what it’s like.  E.g., use df.info(), df.describe().
print(df.info())
print(df.describe())


# 5. We will try to predict whether a college is public or private.  Select a few predictors, 
# create NumPy arrays X and y, and then do a training/test split.  Try hard to remember how to 
# do this from memory.  If you can't, refer to the hints.
predictors = ['Outstate', 'F.Undergrad']
X = df[predictors].values
y = df['Private'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


# 6. Train a tree classifier using Scikit-Learn's DecisionTreeClassifier.  Use the training 
# data you created in the previous step.
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)


# 7. Install graphviz by entering conda install python-graphviz at the Anaconda prompt.  
# Then plot your tree using graphviz.  Try playing with some of the options of export_graphviz(). 
dot_data = export_graphviz(
    clf, 
    precision=2,
    feature_names=predictors,
    proportion=True,
    class_names=['Public', 'Private'],
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("college_tree", format="png", cleanup=True)
graph 


# 8. Use your classification tree to predict whether examples in your test data are public 
# or private.  Compute the confusion matrix and the accuracy of your predictions.
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy:", acc)


# 9. If you still have time, do the following:
# - try building more classification trees, using different sets of input features
# - look at, and play with, the hyperparameters available in DecisionTreeClassifier, especially max_depth.
# - see how much the classification tree that you produce depends on your particular training set
clf2 = DecisionTreeClassifier(max_depth=3, random_state=0)
clf2.fit(X_train, y_train)

dot_data2 = export_graphviz(
    clf2, 
    precision=2,
    feature_names=predictors,
    proportion=True,
    class_names=['Public', 'Private'],
    filled=True,
    rounded=True,
    special_characters=True
)

graph2 = graphviz.Source(dot_data2)
graph2.render("college_tree_depth3", format="png", cleanup=True)
graph2

# Evaluate clf2
y_pred2 = clf2.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred2)
acc2 = accuracy_score(y_test, y_pred2)
print("Confusion Matrix (max_depth=3):\n", cm2)
print("Accuracy (max_depth=3):", acc2)

