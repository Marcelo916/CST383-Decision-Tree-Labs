# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 15:11:18 2025

CST 383 - Lab: Decision Tress Part 2

@author: Marcelo Villalobos Diaz
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 1. We defined the Gini index value for a node in a classification tree as 2 p (1-p)
# Where p is the estimated probability of either of the two classes.  For example, if 
# a node has 30 training instances of class A, and 50 training instances of class B, 
# then the estimated probability of class A is 30/(30 + 50).  What is the Gini index value for this node?
p = 30 / 80
gini_value = 2 * p * (1 - p)
print("Gini for (30,50):", gini_value)


# 2. Create a new Python file.  Fill in the code below to create a function that gives the 
# Gini value for a node in a binary classification tree given values for the number of instances 
# of each class.  class_counts is a list of length two.
def gini(class_counts):
    if sum(class_counts) == 0:
        return 0
    p = class_counts[0] / sum(class_counts)
    return 2 * p * (1 - p)


# 3. Test your function.  What is gini([30, 50])? gini([10, 10])?  What is gini([20, 0])?  
# What is gini([100, 0])? 
print("gini([30, 50]):", gini([30, 50]))    
print("gini([10, 10]):", gini([10, 10]))    
print("gini([20, 0]):", gini([20, 0]))      
print("gini([100, 0]):", gini([100, 0]))


# 4. Add the following code at the top of your file to read and preprocess the data.
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/heart.csv")
df['output'] = df['output'] - 1
df = df[['age', 'maxhr', 'restbp', 'output']]

sns.scatterplot(x='age', y='maxhr', hue='output', data=df)
plt.show()


# 5. Run the code and look at the plot.  If we were going to build a classification tree, 
# and split first on 'age', what do you think a good age value to split on would be?
"""
Good split on age might be around 50.
"""


# 6. Compute the Gini index for df as a whole.  For this you just need the number of rows 
# with output = 0 and the number of rows with output = 1.
gini_root = gini([(df['output'] == i).sum() for i in [0, 1]])
print("Gini index for full dataset:", gini_root)


# 7. Now consider a split on age < 50.  Write code to compute the Gini index for the case 
# of of age < 50 and the Gini index for the case of age >= 50.  For the case of age < 50, 
# get the rows of df where age < 50, then count the number of rows with output = 0 and output = 1.
split_val = 50
df_lo = df[df['age'] < split_val]
df_hi = df[df['age'] >= split_val]

counts_lo = [(df_lo['output'] == i).sum() for i in [0, 1]]
counts_hi = [(df_hi['output'] == i).sum() for i in [0, 1]]

gini_lo = gini(counts_lo)
gini_hi = gini(counts_hi)

print("Gini lo:", gini_lo)
print("Gini hi:", gini_hi)


# 8. Now compute the overall Gini index value for the split on age < 50.  First you need to compute 
# the fraction of nodes associated with age < 50 (call it fraction_lo) and the fraction of nodes 
# associated with age >= 50 (call it fraction_hi).  Then get the Gini value for the split like this 
# (in pseudo code): gini_split = gini_lo * fraction_lo + gini_hi * fraction_hi
# The split is useful if the Gini value for the split is lower than the GIni value for the root.
fraction_lo = df_lo.shape[0] / df.shape[0]
fraction_hi = df_hi.shape[0] / df.shape[0]
gini_split = fraction_lo * gini_lo + fraction_hi * gini_hi
print("Overall Gini for split at age 50:", gini_split)


# 9. Is a split on age < 40 better than a split on age < 50?
split_val_40 = 40
df_lo_40 = df[df['age'] < split_val_40]
df_hi_40 = df[df['age'] >= split_val_40]

counts_lo_40 = [(df_lo_40['output'] == i).sum() for i in [0, 1]]
counts_hi_40 = [(df_hi_40['output'] == i).sum() for i in [0, 1]]

gini_lo_40 = gini(counts_lo_40)
gini_hi_40 = gini(counts_hi_40)

fraction_lo_40 = df_lo_40.shape[0] / df.shape[0]
fraction_hi_40 = df_hi_40.shape[0] / df.shape[0]

gini_split_40 = fraction_lo_40 * gini_lo_40 + fraction_hi_40 * gini_hi_40
print("Overall Gini for split at age 40:", gini_split_40)


# 10. Compute the Gini value for all age splits where age ranges from 20 to 80.  
# Then plot the Gini split value for all the ages (age on x axis, Gini value on y axis).  
# What is the best age value for a split on age?
ages = np.arange(20, 81)
gini_splits = []

for split_val in ages:
    df_lo = df[df['age'] < split_val]
    df_hi = df[df['age'] >= split_val]

    counts_lo = [(df_lo['output'] == i).sum() for i in [0, 1]]
    counts_hi = [(df_hi['output'] == i).sum() for i in [0, 1]]

    gini_lo = gini(counts_lo)
    gini_hi = gini(counts_hi)

    fraction_lo = df_lo.shape[0] / df.shape[0]
    fraction_hi = df_hi.shape[0] / df.shape[0]

    gini_split = fraction_lo * gini_lo + fraction_hi * gini_hi
    gini_splits.append(gini_split)

plt.plot(ages, gini_splits, 'o')
plt.xlabel('age')
plt.ylabel('Gini')
plt.title('Gini index value by age split')
plt.show()

