import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import time

#set seed for reproducibility
SEED = 42
np.random.seed(SEED)

#load df from CSV
expr_df = pd.read_csv("project/data/df_final3.csv", index_col=0)

#keep only gene columns for x (all except known metadata columns)
metadata_col = ['pcr_response']
x = expr_df.drop(columns=metadata_col)
y = expr_df['pcr_response']  #99 1 and 389 0

#Z-score normalization of gene expression columns (column-wise)
x = (x - x.mean()) / x.std()

#define evaluation function
def evaluate_rf(hyperparams):
    n_estimators = int(np.round(hyperparams[0]))
    max_depth = int(np.round(hyperparams[1]))
    min_samples_leaf = int(np.round(hyperparams[2]))
    max_features = float(hyperparams[3])
    #valid ranges
    max_depth = max(1, max_depth)
    min_samples_leaf = max(1, min_samples_leaf)

    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED) #stratified CV for handling class imbalance
    train_auprcs, test_auprcs = [], []

    for train_idx, test_idx in outer_cv.split(x, y):
        x_tr, x_te = x.iloc[train_idx], x.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=32
        )
        model.fit(x_tr, y_tr)

        #calculate performance on both train and test set
        y_train_prob = model.predict_proba(x_tr)[:, 1]
        y_test_prob = model.predict_proba(x_te)[:, 1]

        precision_train, recall_train, _ = precision_recall_curve(y_tr, y_train_prob)
        precision_test, recall_test, _ = precision_recall_curve(y_te, y_test_prob)

        train_auprc = auc(recall_train, precision_train)
        test_auprc = auc(recall_test, precision_test)

        train_auprcs.append(train_auprc)
        test_auprcs.append(test_auprc)

    mean_test = np.mean(test_auprcs)
    mean_train = np.mean(train_auprcs)
    gap = mean_train - mean_test

    return [-mean_test, gap]  #negative AUPRC because pymoo minimizes by default

#define pymoo Problem
class RandomForestOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=4,  #hyperparameters
            n_obj=2,  #objectives
            n_constr=0,
            xl=np.array([20, 1, 1, 0.05]),   #lower bounds
            xu=np.array([100, 8, 15, 0.8])   #upper bounds
        )

    def _evaluate(self, X, out, *args, **kwargs):
        results = np.array([evaluate_rf(x) for x in X])
        out["F"] = results

#create problem instance
problem = RandomForestOptimizationProblem()

algorithm = NSGA2(
    pop_size=40,
    sampling=FloatRandomSampling(),  #random sampling of floats for population initialization
    crossover=SBX(prob=0.9, eta=10), #single point crossover on real valued vectors
    mutation=PM(eta=10),             #polynomial mutation
    eliminate_duplicates=True
)

#run optimization with computation time tracking
termination = get_termination("n_gen", 12)
start_time = time.time()
res = minimize(problem, algorithm, termination, seed=SEED, save_history=True, verbose=True)
end_time = time.time()
elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)
print(f"\nTotal runtime: {minutes} min {seconds} sec")

#round objective values to avoid floating 
rounded_F = np.round(res.F, decimals=4)
#find unique rows in F
_, unique_indices = np.unique(rounded_F, axis=0, return_index=True)
#get unique solutions
unique_F = res.F[unique_indices]
unique_X = res.X[unique_indices]

print("\nFinal Pareto Front:")
for i, (obj, x_sol) in enumerate(zip(unique_F, unique_X)):
    test_auprc = -obj[0]
    gap = obj[1]
    print(f"Solution {i+1}: Test AUPRC = {test_auprc:.4f}, Gap = {gap:.4f}, Params = n_est={x_sol[0]:.0f}, max_depth={x_sol[1]:.0f}, min_leaf={x_sol[2]:.0f}, max_feat={x_sol[3]:.3f}")

#initialize lists for storing history stats
obj1_best, obj1_mean, obj1_worst = [], [], []
obj2_best, obj2_mean, obj2_worst = [], [], []

#extract statistics for each generation and save them
for gen, entry in enumerate(res.history):
    F = entry.pop.get("F")
    if F is None or len(F) == 0:
        obj1_best.append(np.nan)
        obj1_mean.append(np.nan)
        obj1_worst.append(np.nan)
        obj2_best.append(np.nan)
        obj2_mean.append(np.nan)
        obj2_worst.append(np.nan)
    else:
        obj1 = F[:, 0]  #-test AUPRC
        obj2 = F[:, 1]  #gap

        obj1_best.append(np.min(obj1))
        obj1_mean.append(np.mean(obj1))
        obj1_worst.append(np.max(obj1))

        obj2_best.append(np.min(obj2))
        obj2_mean.append(np.mean(obj2))
        obj2_worst.append(np.max(obj2))

history_df = pd.DataFrame({
    "gen": np.arange(1, len(obj1_best) + 1),
    "obj1_best": obj1_best,
    "obj1_mean": obj1_mean,
    "obj1_worst": obj1_worst,
    "obj2_best": obj2_best,
    "obj2_mean": obj2_mean,
    "obj2_worst": obj2_worst,
})

history_df.to_csv("nsga2_fitness_history.csv", index=False)
