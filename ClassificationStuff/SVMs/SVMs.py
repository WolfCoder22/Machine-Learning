import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler


"""
Various Classification Problems

"""

def performLinearSVM()