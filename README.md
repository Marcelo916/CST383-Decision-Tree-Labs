# CST383 – Decision Tree & Model Evaluation Labs

Hands-on labs exploring decision trees (classification & regression), the Gini impurity criterion,
and learning-curve analysis with k-NN.

## Labs
- **decision_trees.py** — Decision tree **classification** on College data; train/test split, confusion matrix & accuracy; exports tree plots with Graphviz.
- **decision_trees_2.py** — Implement the **Gini index** and search best **age** split on Heart data; plot Gini vs. threshold and `age`–`maxhr` scatter.
- **decision_trees_3.py** — Decision tree **regression** on CPU performance; compare small/medium/large trees with **RMSE**.
- **system-design-lab.py** — **k-NN** learning curves on German credit; error vs. training size for k = 1,3,5,9; repeat after adding a categorical feature.

## Run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib seaborn graphviz
# If tree images don't render, install Graphviz system package:
# - Windows: https://graphviz.org/download/ (add to PATH)
# - macOS: brew install graphviz
# - Linux: sudo apt-get install graphviz
python decision_trees.py
