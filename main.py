'''
Author: Cheryl Tse
Date: 2024-08-22 17:23:38
LastEditTime: 2025-06-03 10:40:11
Description: 
'''
import time
import warnings
from collections import Counter

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize

from GAdaBoost.GAdaBoost import GSAMME


def main(Noise_ratio, data_nm, seed):
    data_frame = pd.read_csv(r"/Users/xieqin/Documents/dataset/noise/noise" + str(Noise_ratio) + '/'+ data_nm + str(Noise_ratio) + ".csv",
                             header=None)  
    data = data_frame.values  
    data = np.array(data)
    file.write(str(data.shape) + '\n')
    file.write(str(Counter(data[:, 0])) + '\n')
    minMax = MinMaxScaler()     
    X_data = minMax.fit_transform(data[:, 1:])
    y_data = data[:, 0].astype(int)
    le = LabelEncoder()   
    y_data = le.fit_transform(y_data)   
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=seed)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'random_state': trial.suggest_categorical("random_state", [seed]),
        }
    
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(KFold(n_splits=5, shuffle=True, random_state=seed).split(X_train)):
            try:
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                model = GSAMME(**params)
                model.fit(X_fold_train, y_fold_train)
                score = accuracy_score(y_fold_val, model.predict(X_fold_val))
                cv_scores.append(score)
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                if len(np.unique(y_fold_val)) > 2:  
                    score = f1_score(y_fold_val, y_pred, average='weighted')
                else:  
                    score = f1_score(y_fold_val, y_pred, average='binary', pos_label=0)
                cv_scores.append(score)
            except Exception as e:
                print(f"Fold {fold} failed: {str(e)}")
                return float('nan')
        
        return np.mean(cv_scores)
    
    def find_best_trial_with_min_estimators(study):
        completed_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        max_value = max(t.value for t in completed_trials)
        best_trials = [t for t in completed_trials if t.value == max_value]
        min_max_depth = min(t.params['max_depth'] for t in best_trials)
        best_trials = [t for t in best_trials if t.params['max_depth'] == min_max_depth]
        best_trial = min(best_trials, key=lambda t: t.params['n_estimators'])
        return best_trial

    #optuna.seed(seed)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    file.write('Best trial:' + '\n')
    print('Best trial:')

    trial = find_best_trial_with_min_estimators(study)
    file.write('  Value: '+ str(trial.value) + '\n')
    print('  Value: ', trial.value)
    file.write('  Params: ' + '\n')
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
        file.write('    {}: {}'.format(key, value) + '\n')

    start_time_train = time.time()
    best_params = study.best_params
    #final_model = GSAMME_ablation(**best_params) 
    final_model = GSAMME(**best_params)
    final_model.fit(X_train, y_train)
    end_time_train = time.time()
    best_training_time = end_time_train - start_time_train

    file.write(f"Training time for best parameters: {best_training_time*1000} milliseconds" + '\n')

    start_time_test = time.time()
    y_pred = final_model.predict(X_test)
    end_time_test = time.time()
    best_testing_time = end_time_test - start_time_test
    accuracy = accuracy_score(y_test, y_pred) 
    if len(np.unique(y_test)) > 2:  
        f1 = f1_score(y_test, y_pred, average='weighted')
    else:  
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=0)
    file.write(f"Testing time for best parameters: {best_testing_time*1000} milliseconds" + '\n')
    file.write('-----------------------------------' + '\n')
    file.write(f"Total time for best parameters: {(best_training_time+best_testing_time)*1000} milliseconds" + '\n')
    #print(f'Test accuracy: {accuracy:.4f}')
    file.write(f'Test accuracy: {accuracy:.4f}' + '\n')
    #print(f'Test F1: {f1:.4f}')
    file.write(f'Test F1: {f1:.4f}' + '\n')

if __name__ == "__main__":
    #for nr in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    for nr in [0.2]:
        Noise_ratio = nr   
        seed = 42
        file = open('GSAMME_optuna' + str(seed) +'_'+ str(Noise_ratio) + '.txt', mode='a')
        file.write('Noise_ratio: ' + str(Noise_ratio) + '\n')
        warnings.filterwarnings("ignore") 
        data_list = ['Balancescale','contraceptive','segment','splice','page-blocks', 'svmguide1','thyroid','coil2000','penbased','nursery','shuttle','fars']
        data_list = ['contraceptive']
        for data_nm in data_list:
            print('data name: ', data_nm)
            file.write('data name: ' + data_nm + '\n')
            main(Noise_ratio, data_nm, seed)
            file.write('---------------------------------------------' + '\n')
            file.write('---------------------------------------------' + '\n')
        file.write('all done!!!!!'+ '\n')
        file.close()
