from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import Imputer, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from cleanMyTestData.regressionData import getBasicRegData, getSsinData, getEducationData
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#show plot of parameter weight values from Lasso regression
# L1 Regularization- may want to remove zero weights params for ridge regression
def showLassoParamWeights(X, y, alpha=.4, impStrategy='mean'):

    #get column names
    df_columns=X.columns

    #fill NaNs
    imp =Imputer(strategy=impStrategy)
    X= imp.fit_transform(X)

    #create lasso and normalize
    lasso= Lasso(normalize=True)

    lasso.fit(X, y)

    lasso_coef = lasso.coef_

    # Plot the coefficients
    plt.plot(range(len(df_columns)), lasso_coef)
    plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
    plt.margins(0.02)
    plt.show()


def getNewXfromLassoWeightThresh(X, y, alpha=.4, weightThresh=1, impStrategy='mean'):

    # get column names
    df_columns = X.columns

    # fill NaNs
    imp = Imputer(strategy=impStrategy)
    Xnp = imp.fit_transform(X)

    # create lasso and normalize and fit
    lasso = Lasso(normalize=True)
    lasso.fit(Xnp, y)

    lasso_coefs = lasso.coef_

    #create new X dataframe from weightThresh
    mask= lasso_coefs>=weightThresh

    return X.loc[:, mask]

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