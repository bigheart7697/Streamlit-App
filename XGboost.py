import warnings

import xgboost as xgb
from scikitplot import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'
import shap


class XGBmodel(object):

    def __init__(self, param):
        self.param = param

    def fit(self, X, y):
        clf = xgb.XGBClassifier(random_state=22)
        clf_search = RandomizedSearchCV(estimator=clf, n_iter=50, param_distributions=self.param, scoring='accuracy',
                                        cv=10, n_jobs=4)
        clf_search.fit(X, y)
        self.clf = clf_search.best_estimator_
        return self

    def predict_proba(self, X):
        res_proba = self.clf.predict_proba(X)
        return res_proba

    def predict(self, X):
        res = self.clf.predict(X)
        return res

    def show_test_result(self, X, y):
        result_proba = self.predict_proba(X)
        result_ = self.predict(X)
        roc_plot = metrics.plot_roc(y, result_proba)
        predictions = cross_val_predict(self.clf, X, y)
        confusion_matrix_plot = metrics.plot_confusion_matrix(y, predictions, normalize=True)
        return roc_plot, confusion_matrix_plot

    def precision_recall_f1_visual(self, X, y):
        return classification_report(y, self.predict(X), digits=4, output_dict=True)

