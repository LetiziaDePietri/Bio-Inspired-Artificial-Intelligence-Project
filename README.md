# Multi-Objective Optimization of Random Forest Hyperparameters for Overfitting Mitigation in High-Dimensional Biological Data
Project developed for the Bio-Inspired Artificial Intelligence course held by Professor Giovanni Iacca, University of Trento, AY 2024/2025.  
This project implements Multi-Objective Particle Swarm Optimization (MOPSO) and the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to tune Random Forest hyperparameters, with the goal of simultaneously optimizing two objectives: maximizing test AUPRC and minimizing the overfitting gap. This task is particularly relevant in the context of biomedical research and in building machine learning models where the number of features greatly exceeds the number of samples.

- *data_retrieval.ipynb*: data retrieval from GEO and dataset building
- *grid_search.py*: grid-search for RF parameter tuning with 3-fold CV
- *mopso.py*: MOPSO algorithm implementation
- *nsga2.py*: NSGA-II algorithm implementation
- *plots.ipynb*: plots generation for results analysis
