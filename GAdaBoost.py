'''
Author: Cheryl Tse
Date: 2024-07-01 16:24:06
LastEditTime: 2025-03-19 19:46:35
Description: 
'''

import kd_sdg
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier


class GSAMME(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, max_depth=None, learning_rate=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.alphas = [] 
        self.models = [] 
        self.X = [] 
        self.y = [] 
        self.max_depth = max_depth
        self.different_samples = []
        self.random_state = random_state
        self.learning_rate = learning_rate

    def fit(self, X_train, y_train):
        train = np.column_stack((y_train, X_train))
        self.classes_ = np.unique(y_train)
        self.features_ = X_train.shape[1]
        train_classes = len(self.classes_)
        train_num = len(y_train)
        alpha = int(1.0 / (train_classes-1) * np.sqrt(train_num))  
        GB_List = kd_sdg.generateIGList(train, alpha)

        def get_different_samples(y_true, y_pred, X): 
            return [np.hstack((y_true[i], X[i])) for i in range(len(y_true)) if y_true[i] != y_pred[i]]


        for _ in range(self.n_estimators):

            stump = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            if _ != 0:
                self.X, self.y, stop_flag = self.augmSampling(GB_List)
                if stop_flag == 1:  
                    #print('used sample num:', len(self.y))
                    break
            else:  
                #sampled_data = self.iniSampling(GB_List)
                sampled_data = self.iniSampling(GB_List, samples_per_granule = min(int(0.5*alpha),self.features_))
                self.X = sampled_data[:, 1:]
                self.y = sampled_data[:, 0]

            stump.fit(self.X[:, :-1], self.y)  
            y_pred = stump.predict(self.X[:, :-1])

            
            self.different_samples = get_different_samples(self.y, y_pred, self.X)

            
            err = np.mean(y_pred != self.y)

            
            #n_classes = len(self.classes_)
            alpha = (np.log((1 - err) / (err + 1e-10)) + np.log(train_classes - 1)) * self.learning_rate  

            
            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        n_samples = X.shape[0]
        class_predictions = np.zeros((n_samples, len(self.classes_)))
        
        for alpha, model in zip(self.alphas, self.models):
            predictions = model.predict(X)
            class_predictions += alpha * (predictions[:, np.newaxis] == self.classes_)
        
        return self.classes_[np.argmax(class_predictions, axis=1)]
    
    def predict_proba(self, X):

        n_samples = X.shape[0]
        class_probabilities = np.zeros((n_samples, len(self.classes_)))
        
        for alpha, model in zip(self.alphas, self.models):
            predictions = model.predict(X)
            class_probabilities += alpha * (predictions[:, np.newaxis] == self.classes_)
        
        
        class_probabilities /= np.sum(class_probabilities, axis=1, keepdims=True)
        
        return class_probabilities

    
    def iniSampling(self, GB_List_tmp: list, samples_per_granule=1) -> np.ndarray:
        def extract_sample(gb, indices, serial_number):
            sample = gb.data[indices]
            gb.data = np.delete(gb.data, indices, axis=0)
            gb.distences = np.delete(gb.distences, indices)
            if len(indices) == 1:
                return np.append(sample, serial_number).reshape(1, -1)
            return np.column_stack((sample, np.full(len(indices), serial_number)))
        
        sampled_data = []
        for serial_number, gb in enumerate(GB_List_tmp):
            indices = self.weighted_sampling(gb.distences, samples_per_granule)
            samples = extract_sample(gb, indices, serial_number)
            if samples_per_granule == 1:
                sampled_data.append(samples)
            else:
                sampled_data.extend(samples)
                
        return np.array(sampled_data)
    
    def augmSampling(self, GB_List_tmp):
        def sample_from_gb(gb, index):  
            nonlocal flag    
            if len(gb.data) != 0:
                indice = self.weighted_sampling(gb.distences)
                sample = gb.data[indice]
                gb.data = np.delete(gb.data, indice, axis=0)
                gb.distences = np.delete(gb.distences, indice)
                sample = np.append(sample, index)
                self.X = np.vstack((self.X, sample[1:]))
                self.y = np.append(self.y, sample[0])
            else:
                flag += 1

        flag = 0  
        stop_flag = 0
        if len(self.different_samples) != 0:  
            sampled_granules = set()  
            for row in self.different_samples:
                i = int(row[-1])  
                if i not in sampled_granules:
                    sample_from_gb(GB_List_tmp[i], i)
                    sampled_granules.add(i)
            if flag == len(sampled_granules):  
                stop_flag = 1

        return self.X, self.y, stop_flag
    
    def weighted_sampling(self, distences, num_samples=1):
        sorted_indices = np.argsort(distences)[::-1]
        return sorted_indices[:num_samples]
    
    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth, "random_state":self.random_state}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
