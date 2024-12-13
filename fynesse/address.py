# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

import os

import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
import numpy as np

# Adapted from https://www.statsmodels.org/stable/examples/notebooks/generated/glm.html

def plot_fitted(fitted: np.ndarray, target: np.ndarray):
  fig, ax = plt.subplots(figsize=(6,6))
  ax.scatter(fitted, target, alpha=0.2)
  ax.axline([0, 0], [1, 1], color='r', linestyle='--')
  ax.set_aspect(aspect='equal')
  ax.set_xlabel("Fitted values")
  ax.set_ylabel("Observed values")
  ax.set_title("Model fit plot")

  plt.tight_layout()

def plot_leverage(result: sm.regression.linear_model.OLSResults):
  fig, ax = plt.subplots()
  influence = result.get_influence()
  leverage = influence.hat_matrix_diag
  residual_norm = influence.resid_studentized
  ax.scatter(leverage, residual_norm, alpha=0.2)

  plt.tight_layout()

def plot_cooks_dist(result: sm.regression.linear_model.OLSResults,
                    bottom: float,
                    top: float):
  fig, ax = plt.subplots()
  influence = result.get_influence()
  cooks_dist = influence.cooks_distance[0]
  threshold = 4 / (result.nobs - len(result.params) - 1)
  count = (cooks_dist > threshold).sum()
  print(f"Outlier count: {count}")

  _, s, _ = ax.stem(cooks_dist, markerfmt=',')
  s.set_linewidth(1)
  ax.axhline(threshold, color='C7', linestyle='--', label=f'threshold')

  ax.set_xlabel("Observation index")
  ax.set_ylim(bottom, top)
  ax.set_ylabel("Cook's distance")
  ax.set_title("Cook's distance plot")
  ax.legend()

  plt.tight_layout()

def plot_resid(result: sm.regression.linear_model.OLSResults):
  yhat = result.fittedvalues
  resid = result.resid_pearson
  
  fig, ax = plt.subplots()
  ax.scatter(yhat, resid, alpha=0.2)
  ax.axhline(0, color='r', linestyle='--')

  ax.set_xlabel("Fitted values")
  ax.set_ylabel("Pearson residuals")
  ax.set_title("Residual dependence plot")

  plt.tight_layout()

# Adapted from 
# https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible
class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, alpha: float=None, L1_wt: float=1.0, family=None, fit_intercept: bool=True):
        self.model_class = model_class
        self.alpha = alpha
        self.L1_wt = L1_wt
        self.family = family
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        if self.fit_intercept:
          X = sm.add_constant(X)
        self.model_ = self.model_class(y, X, family=self.family)
        if self.alpha is None:
          self.results_ = self.model_.fit()
        else:
          self.results_ = self.model_.fit_regularized(alpha=self.alpha, L1_wt=self.L1_wt, maxiter=2000, cnvrg_tol=1e-4, refit=True)
        return self
    def predict(self, X):
        if self.fit_intercept:
          X = sm.add_constant(X, has_constant='add')
        return self.results_.predict(X)

def save_results(results, prefix: str, data_dir_path: str="data"):
  results_path = data_dir_path + "/" + prefix
  if not os.path.exists(results_path):
    os.makedirs(results_path)
  for idx, res in enumerate(results):
    res.save(results_path + "/" + str(idx) + ".pickle")

def load_results(prefix: str, data_dir_path: str="data"):
  results_path = data_dir_path + "/" + prefix
  results = []
  for file in os.listdir(results_path):
    result = sm.load(results_path + "/" + file)
    results.append(result)
  return results