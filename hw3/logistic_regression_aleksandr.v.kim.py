# coding=utf-8

import numpy as np
from sklearn.base import BaseEstimator


LR_PARAMS_DICT = {
    'C': 0,
    'random_state': 777,
    'iters': 1000,
    'batch_size': 1000,
    'step': 0.05
}


class MyLogisticRegression(BaseEstimator):
    def __init__(self, C, random_state, iters, batch_size, step):
        self.C = C
        self.random_state = random_state
        self.iters = iters
        self.batch_size = batch_size
        self.step = step

    # будем пользоваться этой функцией для подсчёта <w, x>
    def __predict(self, X):
        return np.dot(X, self.w) + self.w0

    # sklearn нужно, чтобы predict возвращал классы, поэтому оборачиваем наш __predict в это
    def predict(self, X):
        res = self.__predict(X)
        res[res > 0] = 1
        res[res < 0] = 0
        return res

    # производная регуляризатора
    def der_reg(self):
        reg = self.C*self.w
        return reg

    # будем считать стохастический градиент не на одном элементе, а сразу на пачке (чтобы было эффективнее)
    def sigma(self, M):
        return 1/(1+np.exp(-M))

    def der_loss(self, x, y):
        # s.shape == (batch_size, features)
        # y.shape == (batch_size,)

        # считаем производную по каждой координате на каждом объекте
        # der_w = 0
        # der_w0 = 0
        # for i in range(x.shape[0]):
        #     der_w += -y[i]*x[i]*self.sigma(-y[i]*self.__predict(x[i]))
        #     der_w0 += -y[i]*self.sigma(-y[i]*self.__predict(x[i]))

        # # yx = x*y.reshape([-1,1])
        # # sig = self.sigma(- np.dot(yx,self.w) + y * self.w0)
        # # der_w = - yx*sig.reshape([-1,1])
        # # der_w0 = - y * sig

        sig = self.sigma(self.__predict(x))
        der_w = x * (sig - y).reshape([-1,1])
        der_w0 = (sig - y)
        # y[y==0] = -1
        # yx = x * y.reshape([-1, 1])
        # sig = self.sigma(- np.dot(yx,self.w) + y * self.w0)
        # der_w = -yx*sig.reshape([-1,1])
        # der_w0 = -y * sig


        # для масштаба возвращаем средний градиент по пачке
        return np.dot(1/x.shape[0]*np.ones(x.shape[0]),der_w), np.dot(der_w0,1/x.shape[0]*np.ones(x.shape[0]))

    def fit(self, X_train, y_train):
        # RandomState для воспроизводитмости
        random_gen = np.random.RandomState(self.random_state)
        
        # получаем размерности матрицы
        size, dim = X_train.shape
        
        # случайная начальная инициализация
        self.w = random_gen.rand(dim)
        self.w0 = random_gen.randn()

        for _ in range(self.iters):  
            # берём случайный набор элементов
            rand_indices = random_gen.choice(size, self.batch_size)
            # исходные метки классов это 0/1
            x = X_train[rand_indices]
            y = y_train[rand_indices]

            # считаем производные
            der_w, der_w0 = self.der_loss(x, y)
            der_w += self.der_reg()

            # обновляемся по антиградиенту
            self.w -= der_w * self.step
            self.w0 -= der_w0 * self.step

        # метод fit для sklearn должен возвращать self
        return self