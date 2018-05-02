# coding=utf-8
import numpy as np
from sklearn.base import TransformerMixin
from collections import Counter

LR_PARAMS_DICT = {'penalty':'l1',
                  'C':.01,
                  'random_state':180087
                  }

class CustomTransformer(TransformerMixin):

    def fit(self, X, y):
        self.cnt = Counter()
        self.tol = 10**-10
        X_cuted = np.delete(X, [15, 5, 16], 1)
        for i in range(X_cuted.shape[1]):
            for j in set(X_cuted[:, i]):
                indices = X_cuted[:, i] == j
                goods = np.sum(y[indices])
                bads = np.sum([indices]) - goods
                val = np.log((self.tol + goods) / (self.tol + bads))
                self.cnt[(i, j)] = val
        return self

    def transform(self, X):
        X_new = np.delete(X, [15, 5, 16], 1)
        for i in range(X_new.shape[1]):
            for j in set(X_new[:, i]):
                indices = X_new[:, i] == j
                X_new[indices, i] = 1 / (1 + np.exp(-self.cnt[(i, j)])) 
        return X_new
