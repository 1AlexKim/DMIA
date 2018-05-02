from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
# from sklearn import preprocessing
import numpy as np
import os
import imp
import signal
import traceback
import sys
import pandas
import time
start_time = time.time()

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def signal_handler(signum, frame):
    raise Exception("Timed out!")


class Checker(object):
    def __init__(self):
        df = pandas.read_csv('mushrooms.csv', header=None)
        # df.drop([16], axis=1,inplace=True)
        X, y = np.array(df.loc[:, 1:]), np.array(df.loc[:, 0])
        label_encoder = LabelEncoder()
        for i in range(X.shape[1]):
            X[:, i] = label_encoder.fit_transform(X[:, i])
        y = np.equal(y, 'p').astype(int)
        self.X_data, self.y_data = X, y
        self.applications = 0

    def check(self, script_path):
        try:
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(4)
            algo_impl = imp.load_source('transformer_{}'.format(self.applications), script_path)
            self.applications += 1
            try:
                lr_params = algo_impl.LR_PARAMS_DICT
            except AttributeError:
                lr_params = dict()
            pipeline = make_pipeline(
                algo_impl.CustomTransformer(),  #preprocessing.OneHotEncoder(handle_unknown='ignore'),
                LogisticRegression(**lr_params)
            )
            cv = StratifiedKFold(shuffle=True, n_splits=3, random_state=102)
            return cross_val_score(pipeline, self.X_data, self.y_data, cv=cv).mean()
        except:
            traceback.print_exception(*sys.exc_info())
            return None


if __name__ == '__main__':
    print(Checker().check(SCRIPT_DIR + '/transformer_aleksandr.v.kim.py'))
    print(("--- %s seconds ---" % (time.time() - start_time)))