# CST383 – Decision Tree Labs (Classification & Regression)

Hands-on labs from CST 383 focused on building **decision tree** models in Python.
I implemented both **classifiers** and **regressors**, tuned tree depth/pruning to
balance bias/variance, and evaluated models with standard metrics.

## Labs Included
- **decision_trees.py** – basic decision tree **classification**; train/test split, accuracy, confusion matrix.
- **decision_trees_2.py** – **model tuning** (max_depth, min_samples_split), cross-validation, feature importance.
- **decision_trees_3.py** – decision tree **regression**; MAE/MSE evaluation and residual analysis.
- **system-design-lab.py** – small end-to-end workflow: load/clean data → fit tree → evaluate → visualize.

> Some scripts may reference small CSVs or use scikit-learn toy datasets.  
> If a path is needed, update it at the top of the file.

## What I Practiced
- Data cleaning & feature selection
- Train/test split and **k-fold cross-validation**
- Hyperparameter tuning and **pruning**
- Metrics: **accuracy**, **confusion matrix**, **MAE/MSE**
- Interpreting models via **feature importance** and tree plots

## Quickstart
```bash
# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install dependencies
pip install numpy pandas scikit-learn matplotlib

# run a lab
python decision_trees.py
