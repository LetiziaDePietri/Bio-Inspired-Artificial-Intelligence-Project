import pandas as pd
import gzip
import io
import requests
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV 
from sklearn.utils.class_weight import compute_class_weight
import time
import random

#set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

#load df from CSV
expr_df = pd.read_csv("project/data/df_final3.csv", index_col=0)

#keep only gene columns for x (all except known metadata columns)
metadata_col = ['pcr_response']
x = expr_df.drop(columns=metadata_col)
y = expr_df['pcr_response']  #99 1 and 389 0

#Z-score normalization of gene expression columns (column-wise)
x = (x - x.mean()) / x.std()

#training and evaluation
def run_training_with_grid_search(x, y):
    n_splits = 3
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    test_auprc_scores = []
    train_auprc_scores = []

    param_grid = {
        'n_estimators': [20, 40, 60, 80, 100],
        'max_depth': [1, 2, 4, 6, 8],
        'min_samples_leaf': [1, 5, 10, 12, 15],
        'max_features': [0.1, 0.3, 0.5, 0.7, 'sqrt']
    }

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):
        print(f"\nOuter fold {i+1}/{n_splits}")
        x_tr, x_te = x.iloc[train_idx], x.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        base_model = RandomForestClassifier(random_state=42, n_jobs=32, class_weight='balanced')

        #actual grid search
        search = GridSearchCV(estimator=base_model, param_grid=param_grid, scoring='average_precision', cv=inner_cv, n_jobs=32, verbose=0)

        search.fit(x_tr, y_tr)
        best_params = search.best_params_
        print("Best Params:", best_params)

        #retrain best model on full training fold
        model = RandomForestClassifier(**best_params, class_weight='balanced', n_jobs=32, random_state=4)
        model.fit(x_tr, y_tr)

        #AUPRC on train
        y_pred_train = model.predict_proba(x_tr)[:, 1]
        prec_train, rec_train, _ = precision_recall_curve(y_tr, y_pred_train)
        auprc_train = auc(rec_train, prec_train)
        train_auprc_scores.append(auprc_train)
        print(f"Train AUPRC: {auprc_train:.4f}")

        #AUPRC on test
        y_pred_test = model.predict_proba(x_te)[:, 1]
        prec_test, rec_test, _ = precision_recall_curve(y_te, y_pred_test)
        auprc_test = auc(rec_test, prec_test)
        test_auprc_scores.append(auprc_test)
        print(f"Test AUPRC: {auprc_test:.4f}")

    #summary display
    print("\nTest AUPRC scores per fold:")
    for i, score in enumerate(test_auprc_scores, 1):
        print(f"Fold {i}: {score:.4f}")

    mean_test = np.mean(test_auprc_scores)
    mean_train = np.mean(train_auprc_scores)
    gap = mean_train - mean_test

    print(f"\nMean Test AUPRC: {mean_test:.3f} ± {np.std(test_auprc_scores):.3f}")
    print(f"Mean Train AUPRC: {mean_train:.3f} ± {np.std(train_auprc_scores):.3f}")
    print(f"AUPRC Gap (Train - Test): {gap:.3f}")

#keep track of computational time
start_time = time.time()
run_training_with_grid_search(x, y)
end_time = time.time()

elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)

print(f"\nTotal runtime: {minutes} min {seconds} sec")
