from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoLarsIC
from sklearn.preprocessing import Imputer, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from cleanMyTestData.regressionData import getBasicRegData, getSsinData, getEducationData
from RegressionOther import plot_ic_criterion
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

"""
File contains thre different types of Linear Models
    -Lass0 (L1 Penalty), Ridge(L2 Penanlty), and ElasticNet(L2, L1 Combo)

And a BIC/AIC criterion graphing using LassoLarsIC()
    

Methods
    
    performRidgeReg(X, y, cvfolds=5, impStrategy= 'mean', aLow=0, aHigh=1, numAlphas=30)
        - Determines which alpha hyperparamter makes the best RidgeRegression and prints R^2 score
        -Uses Hold out validation 
        -alphas in range alow to aHigh
        -can change impuation strategy from mean
        -Standardizes Data
        
    performLassoReg(X, y, cvfolds=5, impStrategy= 'mean', aLow=0, aHigh=1, numAlphas=30)
        - Determines which alpha hyperparamter makes the best LassoRegression and prints R^2 score
        -Uses Hold out validation 
        -alphas in range alow to aHigh
        -can change impuation strategy from mean
        -Standardizes Data
        
    performElasticReg(X, y, cvfolds=5, impStrategy= 'mean', numRatios=10, aLow=0, aHigh=1, numAlphas=10)
        -Perform ElasticRegression using a combo of L1 and L2 regualarization
        -Uses Hold out validation
        -alphas in range alow to aHigh
        -Number of different ratios to produce from 0-1 in numRatios
        -can change impuation strategy from mean
        -Standardizes Data
        
    testFitvsNumParms(X, y, impStrategy= 'mean')
        -plots a graph with AIC and BIC showing optimal number of paramters with solid line
        -can change impuation strategy from mean
        
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
             ('LassoReg', Lasso() )]

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

#Plot AIC vs BIC
#  -function taken and changed a bit from sci-kit learn documentation
def testFitvsNumParms(X, y, impStrategy='mean'):

    #imputate missing values
    imp= Imputer(missing_values='NaN', strategy=impStrategy, axis=0)
    X= imp.fit_transform(X)

    # normalize data as done by Lars to allow for comparison
    X /= np.sqrt(np.sum(X ** 2, axis=0))

    #LassoLarsIC: least angle regression with BIC / AIC criterion
    model_bic = LassoLarsIC(criterion='bic', normalize=True)
    t1 = time.time()
    model_bic.fit(X, y)
    t_bic = time.time() - t1

    model_aic = LassoLarsIC(criterion='aic')
    model_aic.fit(X, y)

    plt.figure()
    plot_ic_criterion(model_aic, 'AIC', 'b')
    plot_ic_criterion(model_bic, 'BIC', 'r')
    plt.legend()
    plt.title('Information-criterion for model selection (training time %.3fs)'
              % t_bic)

    plt.show()



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
