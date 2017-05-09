import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler


"""
Various Classification Problems

"""


"""
Linear SVM  using SGDClassifier()
    - params
    
    impStrategy
        -mean deafult
    preprocess
        -deafults to StandardScaler()
        
    penalty
        -'l2' deafult
        -'l1'
        - ‘elasticnet’
    loss
        - 'hinge'
        - 'log'- logisitc regression
        - ‘modified_huber'- good if outliers
        - ‘squared_hinge’- quadratcillay penalized 
    
    
    class_weight='None'
        -'balanced'
        custom dictionary of entried
        
        
    hyperParamTesting
        -alpha stuff
        -l1 ratio
    
    chooseFrom Data Analysis
        -penalty
        -loss
    
        
    
        

"""
def performLinearSVM(X, y, impStrategy= 'mean', preprocess=StandardScaler(), penalty='l2', loss='hinge', aLow=0, aHigh=1, numAlphas=30,
                     class_weight='None', numL1Ratios=10):


    # use hold out validation for analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    # create pipeline for Model testing/training
    #penalty, loss, and class weights 
    steps = [('imputation', Imputer(missing_values='NaN', strategy=impStrategy, axis=0)),
             ('scaler', preprocess),
             ('ridgeReg', SGDClassifier(penalty= 'l2', loss=loss, class_weight='None' ))]

    pipeline = Pipeline(steps)

    # create different alpha paramaters to test
    stepsize = (aHigh - aLow) / numAlphas
    alphas = np.arange(aLow, aHigh, stepsize)

    param_grid = {'ridgeReg__alpha': alphas}

    # Create the GridSearchCV
    gm_cv = GridSearchCV(pipeline, param_grid, cv=folds)

    # fit the Grid Search Cross Value Model
    gm_cv.fit(X_train, y_train)

    y_pred = gm_cv.predict(X_test)
    r2 = gm_cv.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)

    print("Best Alpha: " + str(gm_cv.best_params_))
    print("Tuned Ridge Reg R squared: " + str(r2))
    print("Tuned Ridge Reg MSE: " + str(mse))