#################################################################################################
# This script trains an XGBoost classifier model on selected features using skopt for           #
# hyperparameter tuning and k-fold cross validated AUCPR to assess ability of classifier        #
# It carries out hyperparameter tuning using the train dataset only, using GroupKFold           #
# validation to prevent leakge between datasets. The final model is trained using early         #
# stopping, with the pre-split validation set being used here as validation. The full           #
# final model is then re-trained on the full training and valdiation sets, using the            #
# best performing number of rounds                                                              #
#                                                                                               #
# Command line prompt to run script:                                                            #
# python XGBoost_training.py                                                                    #
# -csv_train - csv for pre-split training dataset                                               #
# -csv_val - csv for pre-split validation dataset                                               #
# -num_calls - the number of sets of hyperparameters the Bayesian gp_minimize will evaluate     #                                                                     #
# -kfolds - number of kfolds used by GroupKFold (Value used = 5)                                #
# -tag - tag for model (Example: -tag XGBoost_model_1)                                          #
#################################################################################################


# Importing all packages
import xgboost as xgb
import pandas as pd
from skopt import gp_minimize, dump, load
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import numpy as np
import pickle
import sys
import os
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, roc_curve
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from warnings import filterwarnings
import multiprocessing as mp
import time
import matplotlib.pyplot as plt

# Filter warnings
filterwarnings("ignore")

# Initialize global lists to store the results
train_scores = []
val_scores = []
test_devs = []
tested_params = []
curr_model_hyper_params = ['colsample_bylevel', 'colsample_bytree', 'gamma', 'eta', 'max_delta_step',
                    'max_depth', 'min_child_weight', 'reg_alpha', 'reg_lambda', 'subsample', 'scale_pos_weight', 'objective']


# 1.define hyperparameter space to search
search_space = [
    Real(0.3, 1, name="colsample_bylevel"),
    Real(0.3, 1, name="colsample_bytree"),
    Real(0.01, 1, name="gamma"),
    Real(0.0001, 1, name="eta"),
    Integer(3, 15, name="max_depth"),
    Real(1, 8, name="min_child_weight"),
    Real(0.1, 100, name="reg_alpha"),
    Real(0.1, 100, name="reg_lambda"),
    Real(0.3, 1.0, name="subsample"),
    Integer(5, 15, name='scale_pos_weight'),
    Categorical(['binary:logistic'], name='objective')
    ]

# Class to add progressbar as a callback to hyperparameter search
class tqdm_skopt(object): 
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        self._bar.update()

# 2. Define function to train model with GroupKFold cross validation
def train_booster_with_cv(params, data, kfolds):
    gkf = GroupKFold(n_splits=kfolds)

    train_scores = []
    val_scores = []

    for train_index, val_index in gkf.split(data.drop(['Pose name','PDBCode','Label'], axis=1), data['Label'], groups=data['PDBCode']):
        training_set = data.iloc[train_index]
        val_set = data.iloc[val_index]

        x_train = training_set.drop(['Pose name','PDBCode','Label'], axis=1)
        y_train = training_set['Label'].astype(int)

        x_val = val_set.drop(['Pose name','PDBCode','Label'], axis=1)
        y_val = val_set['Label'].astype(int)

        model = xgb.XGBClassifier(tree_method='gpu_hist', random_state=42, eval_metric='aucpr', **params)
        model.fit(x_train, y_train, early_stopping_rounds=400, eval_set=[(x_train, y_train), (x_val, y_val)], verbose=False, eval_metric='aucpr')

        # Compute final AUCPR scores
        train_scores.append(model.evals_result()['validation_0']['aucpr'][-1])
        val_scores.append(model.evals_result()['validation_1']['aucpr'][-1])

    return np.mean(train_scores), np.mean(val_scores), np.std(train_scores), np.std(val_scores)

# It is used to trainthe model with k-fold cross validation using the provided hyperparameters (params),
# and return the negative mean validation score (g)
# The @use_named_args(space) decorator transforms your objective function so that it can accept parameters as named arguments. 
# Here space is the definition of the hyperparameters to optimize, and each hyperparameter in the space gets mapped to a named 
# argument in the function. Thus, instead of calling objective([0.01, 100]), you can now call 
# objective(learning_rate=0.01, n_estimators=100). This makes the code more readable and intuitive. 
# This is especially helpful when you have a large number of hyperparameters.
@use_named_args(search_space)
def objective(**params):
    mean_train_score, mean_val_score, std_train_score, std_val_score  = train_booster_with_cv(params, dtrain, kfolds)

    # Append the results to the global lists
    train_scores.append(mean_train_score)
    val_scores.append(mean_val_score)
    test_devs.append(std_val_score)
    tested_params.append(params)
    
    # Return mean validation score for optimization (maximize). Instead of trying to maximise the validation score
    # (which skopt can't do directly), we are asking skopt to minimise the negative of the validation score. 
    # This will lead skopt to find the hyperparameters that yield the highest validation score.
    return -mean_val_score

# This function uses gp_minimize, which uses Bayesian optimisation using Gaussian Processes to find the the best set of
# hyperparameters.
def fetch_best_hyperparameters(dtrain, n_calls, kfolds, tag, search_space):
    print(f'Searching hyperparameter space for booster...\n')
    res_gp = gp_minimize(objective, search_space, random_state=42, n_calls=n_calls, callback=tqdm_skopt(total=n_calls, desc="Hyperparameter search"))
    return res_gp


# Parse CLI user inputs.
def parse_args(args): 
    csv_train_path = args[args.index('-csv_train') + 1]
    csv_val_path = args[args.index('-csv_val') + 1]
    num_calls =  int(args[args.index('-num_calls') + 1])
    kfolds = int(args[args.index('-kfolds') + 1])
    tag = args[args.index('-tag') + 1]

    return csv_train_path, csv_val_path, num_calls, kfolds, tag


# Start timer for script runtime.
start_time = time.time()

if __name__ == '__main__': # run script using CLI

    csv_train_path, csv_val_path, num_calls, kfolds, tag = parse_args(sys.argv)
    
    # Make directory output directory if it doesn't exist, using user-defined "tag" to name the directory
    os.makedirs(f"{tag}_boosters/results", exist_ok=True)

    # Load training dataset and validation dataset
    dtrain = pd.read_csv(csv_train_path)
    dval = pd.read_csv(csv_val_path)
    
    
    # Shuffle the training data
    dtrain = dtrain.sample(frac=1, random_state=42).reset_index(drop=True)

    # Fetch best hyperparameters found by gp_minimize
    res_gp = fetch_best_hyperparameters(dtrain, num_calls, kfolds, tag, search_space) 
    
    
    # Train final model with best hyperparameters on training set and validation set for early stopping.
    param_values = res_gp.x
    param_names = [param.name for param in search_space]
    params_dict = dict(zip(param_names, param_values))
    x_train = dtrain.drop(['Pose name','PDBCode','Label'], axis=1)
    y_train = dtrain['Label'].astype(int)
    model = xgb.XGBClassifier(tree_method='gpu_hist', random_state=42, n_estimators=20000, eval_metric='aucpr', **params_dict)
    
    # First, train with early stopping to find the best iteration.
    model.fit(x_train, y_train, early_stopping_rounds=400, eval_set=[(x_train, y_train), (dval.drop(['Pose name','PDBCode','Label'], axis=1), dval['Label'].astype(int))], verbose=False)
    best_iteration = model.best_iteration
    
    evals_result_final = {}
    evals_result_final = model.evals_result()
    
    
    # Evaluate final model on the training and validation data before it is trained on validation data.
    train_preds = model.predict_proba(x_train)[:, 1]
    val_preds = model.predict_proba(dval.drop(['Pose name','PDBCode','Label'], axis=1))[:, 1]

    train_aucpr = average_precision_score(y_train, train_preds)
    val_aucpr = average_precision_score(dval['Label'].astype(int), val_preds)

    print(f"Final training AUCPR: {train_aucpr}")
    print(f"Final validation AUCPR: {val_aucpr}")

    # Crude overfitting warning - learning curve is more informative.
    if train_aucpr > val_aucpr:
        print("Warning: model may be overfitting!")
    
    # Save AUCPR for validation and training set to file.
    with open(f"{tag}_boosters/overtfitting_results.txt", "w") as f:
        f.write(f"Final training AUCPR: {train_aucpr} \n Final validation AUCPR: {val_aucpr}")
    
    # Then, retrain the model on the full dataset (training and validation sets) for the optimal number of rounds.
    model = xgb.XGBClassifier(tree_method='gpu_hist', random_state=42, n_estimators=best_iteration, eval_metric='aucpr', **params_dict)
    model.fit(pd.concat([x_train, dval.drop(['Pose name','PDBCode','Label'], axis=1)]), pd.concat([y_train, dval['Label'].astype(int)]), eval_metric='aucpr', verbose=False)
    
    # Save final model to file.
    pickle.dump(model, open(f"{tag}_boosters/final_model.pkl", "wb"))
    
    
    # Plot learning curve for the final model.
    plt.figure(figsize=(10, 5))
    plt.plot(evals_result_final['validation_0']['aucpr'], label='Training')
    plt.plot(evals_result_final['validation_1']['aucpr'], label='Validation')
    plt.xlabel('Round')
    plt.ylabel('AUCPR')
    plt.title(f"{tag}_booster Learning Curve")
    plt.legend()
    plt.savefig(f"{tag}_boosters/results/learning_curve.png")
    plt.close()        
        
        
    # Save hyperparameters and scores to file.
    search_results = pd.DataFrame({
        'hyperparameters': tested_params,
        'mean_train_score': train_scores,
        'mean_val_score': val_scores,
        'std_train_score': [np.std(score) for score in train_scores],
        'std_val_score': test_devs,
    })
    search_results.to_csv(f'{tag}_boosters/results/search_results.csv', index=False)

    # Calculate total runtime and write it to an output file.
    total_time = time.time() - start_time
    with open(f'/home/s2451611/MScProject/{tag}_training_runtime.txt', 'w') as f:
        f.write(f'Total runtime: {total_time} seconds')