import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoLarsIC
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler

"""
File contains: 

1.Three different types of Linear Models
    -Lass0 (L1 Penalty), Ridge(L2 Penanlty), and ElasticNet(L2, L1 Combo)

2. BIC/AIC criterion graphing using LassoLarsIC()

3. Graphing weight values from a LassoRegression Model

4. Function to get a new Pandas DF from a Lasso weight Threshold
    

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
    
    #taken, renamed, and briefly edited from sci-kit learn Documentation
    testFitvsNumParms(X, y, impStrategy= 'mean')
        -plots a graph with AIC and BIC showing optimal number of paramters with solid line
        -can change impuation strategy from mean
        
    showLassoParamWeights(X, y, alpha=.4, impStrategy='mean')
        -used to show which weights go to zero from Lasso
        -can change alpha: should test optimal first from performLassoReg()
        -try removing the parameters with small weights from this graph in a Ridge Regression
    
    getNewXfromLassoWeightThresh(X, y, alpha=.4, weightThresh=1, impStrategy='mean')
        -Get new Pandas Df of X from a paramter weight Threshold in Lasso
        - Should look at graph from showLassoParamWeights() to determine the threshold
        
"""

def performRidgeReg(X, y, folds=5, impStrategy= 'mean', aLow=0, aHigh=1, numAlphas=10):

    #use hold out validation for analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    #create pipeline for Model testing/training
    steps = [('imputation', Imputer(missing_values='NaN', strategy= impStrategy, axis=0)),
             ('scaler', StandardScaler()),
             ('ridgeReg', Ridge(random_state=2))]

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


def performLassoReg(X, y, folds=5, impStrategy= 'mean', aLow=0, aHigh=1, numAlphas=10):

    #use hold out validation for analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


    #create pipeline for Model testing/training
    steps = [('imputation', Imputer(missing_values='NaN', strategy= impStrategy, axis=0)),
             ('scaler', StandardScaler()),
             ('LassoReg', Lasso(random_state=2) )]

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
             ('elasticnet', ElasticNet(random_state=2))]

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
