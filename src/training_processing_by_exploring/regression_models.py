from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge, ARDRegression, TweedieRegressor, PoissonRegressor, GammaRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, RANSACRegressor, HuberRegressor, TheilSenRegressor, QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso, LassoLars, LassoLarsIC, Lars
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import OrthogonalMatchingPursuit

from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

class regression_model(object):

    def __init__(
            self,
            X_train, 
            y_train, 
            X_test, 
            y_test):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.scores = ['max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'neg_root_mean_squared_error', 'r2']
        self.keys = ['fit_time', 'score_time', 'test_max_error', 'test_neg_mean_absolute_error', 'test_neg_mean_squared_error', 'test_neg_median_absolute_error', 'test_neg_root_mean_squared_error', 'test_r2']

        self.status = None
        self.response_training = None

    #function to obtain metrics using the testing dataset
    def get_performances(self, description, predict_label, real_label):
        r2_value = r2_score(real_label, predict_label)
        mean_abs_error_value = mean_absolute_error(real_label, predict_label)
        mean_square_error_value = mean_squared_error(real_label, predict_label, squared=False)
        
        row = [description, r2_value, mean_abs_error_value, mean_square_error_value]
        return row

    #function to process average performance in cross val training process
    def process_performance_cross_val(self, performances):
        
        row_response = []
        for i in range(len(self.keys)):
            value = np.mean(performances[self.keys[i]])
            row_response.append(value)
        return row_response

    #function to train a predictive model
    def training_process(self, model, cv_value, description):
        print("Train model with cross validation")
        model.fit(self.X_train, self.y_train)
        response_cv = cross_validate(model, self.X_train, self.y_train, cv=cv_value, scoring=self.scores)
        performances_cv = self.process_performance_cross_val(response_cv)

        print("Predict responses and make evaluation")
        responses_prediction = model.predict(self.X_test)
        response = self.get_performances(description, responses_prediction, self.y_test)
        response = response + performances_cv
        return response

    def make_exploration(self):
        matrix_data = []
        k_fold_value = 5
        print("Start exploring")

        try:
            print("Train KernelRidge")
            rgx_model = KernelRidge()
            response = self.training_process(rgx_model, k_fold_value, "KernelRidge")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train GaussianProcessRegressor")
            rgx_model = GaussianProcessRegressor()
            response = self.training_process(rgx_model, k_fold_value, "GaussianProcessRegressor")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train BayesianRidge")
            rgx_model = BayesianRidge()
            response = self.training_process(rgx_model, k_fold_value, "BayesianRidge")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train GammaRegressor")
            rgx_model = GammaRegressor()
            response = self.training_process(rgx_model, k_fold_value, "GammaRegressor")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train TweedieRegressor")
            rgx_model = TweedieRegressor()
            response = self.training_process(rgx_model, k_fold_value, "TweedieRegressor")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train PoissonRegressor")
            rgx_model = PoissonRegressor()
            response = self.training_process(rgx_model, k_fold_value, "PoissonRegressor")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train SGDRegressor")
            rgx_model = SGDRegressor()
            response = self.training_process(rgx_model, k_fold_value, "SGDRegressor")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train PassiveAggressiveRegressor")
            rgx_model = PassiveAggressiveRegressor()
            response = self.training_process(rgx_model, k_fold_value, "PassiveAggressiveRegressor")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train GradientBoostingRegressor")
            rgx_model = GradientBoostingRegressor()
            response = self.training_process(rgx_model, k_fold_value, "GradientBoostingRegressor")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train HistGradientBoostingRegressor")
            rgx_model = HistGradientBoostingRegressor()
            response = self.training_process(rgx_model, k_fold_value, "HistGradientBoostingRegressor")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train AdaBoostRegressor")
            rgx_model = AdaBoostRegressor()
            response = self.training_process(rgx_model, k_fold_value, "AdaBoostRegressor")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train RandomForestRegressor")
            rgx_model = RandomForestRegressor()
            response = self.training_process(rgx_model, k_fold_value, "RandomForestRegressor")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train ExtraTreesRegressor")
            rgx_model = ExtraTreesRegressor()
            response = self.training_process(rgx_model, k_fold_value, "ExtraTreesRegressor")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train BaggingRegressor")
            rgx_model = BaggingRegressor()
            response = self.training_process(rgx_model, k_fold_value, "BaggingRegressor")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train DecisionTreeRegressor")
            rgx_model = DecisionTreeRegressor()
            response = self.training_process(rgx_model, k_fold_value, "DecisionTreeRegressor")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train SVR")
            rgx_model = SVR()
            response = self.training_process(rgx_model, k_fold_value, "SVR")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train KNeighborsRegressor")
            rgx_model = KNeighborsRegressor()
            response = self.training_process(rgx_model, k_fold_value, "KNeighborsRegressor")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train PLSCanonical")
            rgx_model = PLSCanonical()
            response = self.training_process(rgx_model, k_fold_value, "PLSCanonical")
            matrix_data.append(response)
        except:
            pass

        try:
            print("Train PLSRegression")
            rgx_model = PLSRegression()
            response = self.training_process(rgx_model, k_fold_value, "PLSRegression")
            matrix_data.append(response)
        except:
            pass

        if len(matrix_data)>0:
            self.response_training = pd.DataFrame(matrix_data, columns=['description', 'r2_value', 'mean_abs_error_value', 'mean_square_error_value', 'fit_time', 'score_time', 'test_max_error', 'test_neg_mean_absolute_error', 'test_neg_mean_squared_error', 'test_neg_median_absolute_error', 'test_neg_root_mean_squared_error', 'test_r2'])
            self.status="OK"
        else:
            self.status="ERROR"
