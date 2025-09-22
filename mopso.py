import numpy as np
import pandas as pd
import gzip
import io
import requests
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc
import random
import time

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

#define the MOPSO particle
class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position.copy()
        self.best_score = None

#evaluation function with 2 objectives
def evaluate(position, x, y):
    #convert continuous particle positions to actual hyperparameter values
    n_estimators = int(round(position[0]))
    max_depth = int(round(position[1]))
    min_samples_leaf = int(round(position[2]))
    max_features = float(position[3])

    n_splits = 3 #CV for proper model performance evaluation
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)  #stratified handles class imbalance 
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
            random_state=42,
            n_jobs=32
        )
        model.fit(x_tr, y_tr)

        #calculate model performance on both train and test set
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

    #return objectives to minimize: [-test_auprc, gap]
    return -mean_test, gap

#function to assess particle domination
def dominates(score_a, score_b, epsilon=1e-6):
    """
    Returns True if score_a dominates score_b with epsilon tolerance
    """
    #A dominates B if A is better in all objectives AND strictly better in at least one
    better_or_equal = all(a <= b + epsilon for a, b in zip(score_a, score_b))
    strictly_better = any(a < b - epsilon for a, b in zip(score_a, score_b))
    
    return better_or_equal and strictly_better

#maintain an archive of non-dominated solutions
def update_archive(archive, new_particle, epsilon=1e-6):
    """
    Update archive with new particle, maintaining only non-dominated solutions
    Uses epsilon-dominance to allow more solutions in the archive
    """
    new_score = new_particle['score']
    new_position = new_particle['position']

    #check for exact duplicate in position
    for existing in archive:
        if np.allclose(existing['position'], new_position, atol=0, rtol=0):
            return archive  #skip adding duplicate
    
    #check if new particle is dominated by any existing solution
    dominated = False
    for existing in archive:
        if dominates(existing['score'], new_score, epsilon):
            dominated = True
            break
    
    #do not add the new particle if dominated
    if dominated:
        return archive
    
    #remove all solutions dominated by the new particle
    non_dominated = []
    for existing in archive:
        if not dominates(new_score, existing['score'], epsilon):
            non_dominated.append(existing)
    
    #add the new particle
    non_dominated.append(new_particle)
    
    return non_dominated

def personal_dominates(score_a, score_b, epsilon=1e-6):
    """
    For personal best update with epsilon tolerance
    """
    return dominates(score_a, score_b, epsilon)

#leader selection for social learning based on crowding distance: promotes diversity
def select_diverse_leader(archive, current_particle_pos):
    """
    Select leader using crowding distance to promote diversity
    """
    if len(archive) <= 1:
        return random.choice(archive)['position'] if archive else current_particle_pos
    
    #calculate crowding distances
    archive_with_distance = []
    for sol in archive:
        distance = 0
        #calculate distance to all other solutions
        for other in archive:
            if sol is not other:
                #euclidean distance in objective space
                obj_dist = np.sqrt(sum((a - b)**2 for a, b in zip(sol['score'], other['score'])))
                distance += 1.0 / (obj_dist + 1e-10)
        archive_with_distance.append((sol, 1.0 / (distance + 1e-10)))
    
    #select based on crowding distance (prefer less crowded areas)
    weights = [dist for _, dist in archive_with_distance]
    selected = random.choices(archive_with_distance, weights=weights)[0][0]
    return selected['position']

def adaptive_parameters(generation, max_generations):
    """
    Adaptive parameters to balance exploration and exploitation
    """
    #decrease inertia weight over time
    w_max, w_min = 0.9, 0.1
    w = w_max - (w_max - w_min) * generation / max_generations
    
    #increase social component over time, decrease cognitive
    c1 = 2.5 - 1.5 * generation / max_generations  #2.5 -> 1.0
    c2 = 0.5 + 1.5 * generation / max_generations  #0.5 -> 2.0
    
    return w, c1, c2

def mutation(particle, bounds, mutation_rate=0.25, mutation_strength=0.15): #20% chance to mutate a particle
    """
    Apply mutation to maintain diversity - increased rates for more exploration
    """
    if random.random() < mutation_rate:   #mutation chance per particle
        for i in range(len(particle.position)):
            if random.random() < 0.4:  #if mutating particle, 40% chance to mutate each dimension
                low, high = bounds[i]
                #add random noise proportional to parameter range
                noise = random.gauss(0, mutation_strength * (high - low))  #mutation noise with mean 0 and sd adapted to the size of the parameter range
                particle.position[i] = np.clip(particle.position[i] + noise, low, high)

#initialize lists to store stats
history_obj1_best, history_obj1_mean, history_obj1_worst = [], [], []
history_obj2_best, history_obj2_mean, history_obj2_worst = [], [], []

#MOPSO main loop
def mopso_enhanced(x, y, bounds, n_particles=50, n_iterations=20, archive_size_limit=30):
    particles = [Particle(bounds) for _ in range(n_particles)]
    archive = []

    #initialize particles with Latin Hypercube Sampling for better coverage    
    n_dims = len(bounds)
    intervals = np.linspace(0, 1, n_particles + 1)
    
    for i, p in enumerate(particles):
        #use LHS for better initial distribution
        for j in range(n_dims):
            low, high = bounds[j]
            #sample from the i-th interval for dimension j
            interval_start = intervals[i]
            interval_end = intervals[i + 1]
            uniform_sample = random.uniform(interval_start, interval_end)
            p.position[j] = low + uniform_sample * (high - low)
        
        #add some randomness to avoid too regular patterns
        for j in range(n_dims):
            low, high = bounds[j]
            noise = random.gauss(0, 0.1 * (high - low))
            p.position[j] = np.clip(p.position[j] + noise, low, high)
        
        #particle evaluation
        score = evaluate(p.position, x, y)
        p.best_score = score
        particle_data = {'position': p.position.copy(), 'score': score}
        archive = update_archive(archive, particle_data, epsilon=1e-5)
        
        if i % 10 == 0:
            print(f"Particle {i+1}: Test AUPRC: {-score[0]:.4f}, Gap: {score[1]:.4f}")

    print(f"Initial archive size: {len(archive)}")

    for gen in range(n_iterations):
        print(f"\nGeneration {gen+1}/{n_iterations}")
        
        #get adaptive parameters
        w, c1, c2 = adaptive_parameters(gen, n_iterations)
        print(f"  Parameters: w={w:.3f}, c1={c1:.3f}, c2={c2:.3f}")
        
        for i, p in enumerate(particles):
            #select diverse leader
            if archive:
                leader = select_diverse_leader(archive, p.position)
            else:
                leader = p.best_position
                
            #update velocity with adaptive parameters
            r1, r2 = np.random.rand(len(p.position)), np.random.rand(len(p.position))
            p.velocity = (w * p.velocity + 
                         c1 * r1 * (p.best_position - p.position) + 
                         c2 * r2 * (leader - p.position))
            
            #update position
            p.position += p.velocity

            #keep within bounds
            for j in range(len(p.position)):
                low, high = bounds[j]
                p.position[j] = np.clip(p.position[j], low, high)

            #apply mutation for diversity
            mutation(p, bounds, mutation_rate=0.25, mutation_strength=0.2)

            #evaluate new position
            score = evaluate(p.position, x, y)

            #update personal best
            if p.best_score is None:
                p.best_position = p.position.copy()
                p.best_score = score
            elif personal_dominates(score, p.best_score, epsilon=1e-5):
                p.best_position = p.position.copy()
                p.best_score = score
            elif random.random() < 0.2:  # 20% chance to accept non-dominating solution
                p.best_position = p.position.copy()
                p.best_score = score

            #update archive
            particle_data = {'position': p.position.copy(), 'score': score}
            archive = update_archive(archive, particle_data, epsilon=1e-5)

        #extract objectives of all solutions in archive
        obj_values = np.array([sol['score'] for sol in archive])  #shape (n_solutions, 2)

        if len(obj_values) > 0:
            #for objective 1 (negative test AUPRC)
            obj1 = obj_values[:, 0]
            history_obj1_best.append(np.min(obj1))    #since we minimize -test AUPRC, min = best
            history_obj1_mean.append(np.mean(obj1))
            history_obj1_worst.append(np.max(obj1))

            #for objective 2 (gap)
            obj2 = obj_values[:, 1]
            history_obj2_best.append(np.min(obj2))
            history_obj2_mean.append(np.mean(obj2))
            history_obj2_worst.append(np.max(obj2))
        else:
            #if no archive solutions append nan
            history_obj1_best.append(np.nan)
            history_obj1_mean.append(np.nan)
            history_obj1_worst.append(np.nan)
            history_obj2_best.append(np.nan)
            history_obj2_mean.append(np.nan)
            history_obj2_worst.append(np.nan)

        print(f"Archive size after generation {gen+1}: {len(archive)}")
        
        #print current best solutions
        if len(archive) > 1:
            sorted_archive = sorted(archive, key=lambda x: -x['score'][0])
            print(f"  Best Test AUPRC: {-sorted_archive[0]['score'][0]:.4f}")
            print(f"  Best Gap: {min(sol['score'][1] for sol in archive):.4f}")
            print(f"  Worst Test AUPRC: {-sorted_archive[-1]['score'][0]:.4f}")
            print(f"  Range of gaps: {min(sol['score'][1] for sol in archive):.4f} - {max(sol['score'][1] for sol in archive):.4f}")

    return archive, (history_obj1_best, history_obj1_mean, history_obj1_worst, history_obj2_best, history_obj2_mean, history_obj2_worst)

#bounds definition
bounds = [
    (20, 100),   #n_estimators
    (1, 8),      #max_depth
    (1, 15),     #min_samples_leaf
    (0.05, 0.8), #max_features
]

#run MOPSO with computational time tracking
start_time = time.time()
results = mopso_enhanced(x, y, bounds, n_particles=40, n_iterations=12, archive_size_limit=20)
end_time = time.time()
elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)
print(f"\nTotal runtime: {minutes} min {seconds} sec")

#sort results by test AUPRC for better visualization
results_sorted = sorted(results[0], key=lambda x: -x['score'][0])

print(f"\nFinal Pareto Front ({len(results_sorted)} solutions):")
for i, r in enumerate(results_sorted):
    test_auprc = -r['score'][0]
    gap = r['score'][1]
    params = r['position']
    print(f"Solution {i+1}:")
    print(f"  Test AUPRC: {test_auprc:.4f}")
    print(f"  Overfitting Gap: {gap:.4f}")
    print(f"  Parameters: n_est={params[0]:.1f}, max_depth={params[1]:.1f}, min_samples_leaf={params[2]:.1f}, max_features={params[3]:.3f}")
    print()

#additional analysis
print(f"\nPareto Front Analysis:")
print(f"Test AUPRC range: {min(-sol['score'][0] for sol in results_sorted):.4f} - {max(-sol['score'][0] for sol in results_sorted):.4f}")
print(f"Gap range: {min(sol['score'][1] for sol in results_sorted):.4f} - {max(sol['score'][1] for sol in results_sorted):.4f}")

#save fitness landscape stats
(obj1_best, obj1_mean, obj1_worst, obj2_best, obj2_mean, obj2_worst) = results[1]

history_df = pd.DataFrame({
    "gen": np.arange(1, len(obj1_best) + 1),
    "obj1_best": obj1_best,
    "obj1_mean": obj1_mean,
    "obj1_worst": obj1_worst,
    "obj2_best": obj2_best,
    "obj2_mean": obj2_mean,
    "obj2_worst": obj2_worst,
})

history_df.to_csv("mopso_fitness_history.csv", index=False)
