from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import Imputer, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from cleanMyTestData.regressionData import getBasicRegData, getSsinData, getEducationData
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
File contains thre different types of Linear Models
-Lass0 (L1 Penalty), Ridge(L2 Penanlty), and ElasticNet(L2, L1 Combo)

Methods
    
    performRidgeReg(X, y, folds=5, impSrtat= 'mean')
        - determines which alpha hyperparamter makes the best ridge regression and prints R^2 score

"""

def performRidgeReg(X, y, folds=5, impStrategy= 'mean', aLow=0, aHigh=1, numAlphas=30):

    #use hold out validation for analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    #create pipeline for Model testing/training
    steps = [('imputation', Imputer(missing_values='NaN', strategy= impStrategy, axis=0)),
             ('scaler', StandardScaler()),
             ('ridgeReg', Ridge())]

    pipeline = Pipeline(steps)

    #create different alpha paramaters to test
    stepsize = (aHigh - aLow) / numAlphas
    alphas = np.arange(aLow, aHigh, stepsize)

    param_grid = {'ridgeReg__alpha': alphas}

    # Create the GridSearchCV
    gm_cv = GridSearchCV(pipeline, param_grid, cv=folds)

    #fit the Grid Search Cross Value Model
    gm_cv.fit(X_train, y_train)

    y_pred = gm_cv.predict(X_test)
    r2 = gm_cv.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)

    print("Best Alpha: "+str(gm_cv.best_params_))
    print("Tuned Ridge Reg R squared: "+str(r2))
    print("Tuned Ridge Reg MSE: "+str(mse))


def performLassoReg(X, y, folds=5, impStrategy= 'mean', aLow=0, aHigh=1, numAlphas=30):

    #use hold out validation for analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    #create pipeline for Model testing/training
    steps = [('imputation', Imputer(missing_values='NaN', strategy= impStrategy, axis=0)),
             ('scaler', StandardScaler()),
             ('LassoReg', Lasso())]

    pipeline = Pipeline(steps)

    #create different alpha paramaters to test
    stepsize = (aHigh - aLow) / numAlphas
    alphas = np.arange(aLow, aHigh, stepsize)

    param_grid = {'LassoReg__alpha': alphas}

    # Create the GridSearchCV
    gm_cv = GridSearchCV(pipeline, param_grid, cv=folds)

    #fit the Grid Search Cross Value Model
    gm_cv.fit(X_train, y_train)

    y_pred = gm_cv.predict(X_test)
    r2 = gm_cv.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)

    print("Best Alpha: "+str(gm_cv.best_params_))
    print("Tuned Lasso Reg R squared: "+str(r2))
    print("Tuned Lasso Reg MSE: "+str(mse))


#perform
def performElasticReg(X, y, folds=5, impStrategy= 'mean', numRatios=10, aLow=0, aHigh=1, numAlphas=10):

    #use hold out validation for analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    #create pipeline for Model testing/training
    steps = [('imputation', Imputer(missing_values='NaN', strategy= impStrategy, axis=0)),
             ('scaler', StandardScaler()),
             ('elasticnet', ElasticNet())]

    pipeline = Pipeline(steps)

    #create different alpha and ratio paramaters to test
    stepsize= 1/numRatios
    l1_ratios = np.arange(0, 1, stepsize)

    stepsize = (aHigh - aLow) / numAlphas
    alphas = np.arange(aLow, aHigh, stepsize)

    parameters = {'elasticnet__l1_ratio': l1_ratios,
                  'elasticnet__alpha': alphas}

    # Create the GridSearchCV
    gm_cv = GridSearchCV(pipeline, parameters, cv=folds)

    #fit the Grid Search Cross Value Model
    gm_cv.fit(X_train, y_train)

    y_pred = gm_cv.predict(X_test)
    r2 = gm_cv.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)

    print("Best L1 Ratio: "+str(gm_cv.best_params_))
    print("Tuned Elastic Net R squared: "+str(r2))
    print("Tuned Elastic Net MSE: "+str(mse))



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



X, y= getEducationData()
performElasticReg(X, y)


"""
Notes of model Selection for regression

Lasso (L1 abs val regualizaion)
    - good for achieving sparsity
    -difficult to avoid overfitting
    -good for regression feature selection
    


Ridge   (L1- square paramters)
    -Normally better bias/variance tradeoff
    -Good if Normal Prior distribution
    -Good for High dimensional Data

    -Much Better for Highly correlated features
        do correlation plot
        
    -Avoids overfitting more 
    
    Hyperparamters
        -Alpha
            must be postive float
            -Larger Values specify stronger regualriztation
                -'underfitting'
        
Elastic
    - shrinkage and automatic variable reduction
    
How to check
-Check feature correlation useing plot of print in DF
-Check dimesionalility of data
-Scaling doesn't make much of a difference since an affince map transformation

-Use lAsso Ridge to plot feature importance in Regression
- Remove Features and peform ridege regression


"""
