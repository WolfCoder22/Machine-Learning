from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import Imputer, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from cleanMyTestData.regressionData import getBasicRegData, getSsinData, getEducationData
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#plots AIC vs BIC
#   -taken from Sci-kit learn API
def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')