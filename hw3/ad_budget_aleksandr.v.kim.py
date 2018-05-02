#coding=utf-8

from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from scipy.optimize import minimize, linprog
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

class Optimizer:
    def __init__(self):
        pass

    def optimize(self, origin_budget):
        default_target = self.model.predict([origin_budget])[0]
        default_weights = self.model.coef_
        L = origin_budget * 0.95
        R = origin_budget * 1.05
        c = np.ones(origin_budget.shape[0])
        opt_budget = linprog(c=c,A_ub=-default_weights,b_ub=-default_target,bounds=list(zip(L,R)))

        return opt_budget.x

    def fit(self, X_data, y_data):
        my_cv = LeaveOneOut()
        param_grid = {'alpha':[0.001,0.01,0.1,0.25,0.5,1,2.5,5,10,50,100]}
        sgd_params = {'loss':['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                      'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                      'alpha':[0.0001,0.0005,0.001,0.01,0.1,0.9]}
        sgd = SGDRegressor(random_state=240,fit_intercept=False)
        lasso = Lasso(random_state=240,fit_intercept=False)
        ridge = Ridge(random_state=240,fit_intercept=False)
        grid_search_lasso = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=my_cv).fit(X_data, y_data)
        grid_search_ridge = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=my_cv).fit(X_data, y_data)

        grid_search_sgd = GridSearchCV(estimator=sgd, param_grid=sgd_params, cv=my_cv).fit(X_data, y_data)
        if (grid_search_lasso.best_score_ > grid_search_ridge.best_score_)and(grid_search_lasso.best_score_ > grid_search_sgd.best_score_):
            self.model = grid_search_lasso.best_estimator_.fit(X_data, y_data.astype(int))
        elif (grid_search_ridge.best_score_ > grid_search_lasso.best_score_)and(grid_search_ridge.best_score_ > grid_search_sgd.best_score_):
            self.model = grid_search_ridge.best_estimator_.fit(X_data, y_data.astype(int))
        else:
            self.model = grid_search_sgd.best_estimator_.fit(X_data, y_data.astype(int))