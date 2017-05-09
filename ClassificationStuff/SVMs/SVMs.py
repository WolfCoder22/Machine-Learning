import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from ClassificationStuff.classificationData import getWineData


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
def performLinearSVM(X, y, impStrategy= 'mean', preprocess=StandardScaler(), folds=5, penalty='l2', loss='hinge', aLow = 0.0001, aHigh=.1, numAlphas=10,
                     class_weight=None, l1RatLow=0, l1RatHigh=.5, numL1Ratios=10):


    # use hold out validation for analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    # create pipeline for Model testing/training
    #penalty, loss, and class weights
    steps = [('imputation', Imputer(missing_values='NaN', strategy=impStrategy, axis=0)),
             ('scaler', preprocess),
             ('kernelSVM', SGDClassifier(kernel=kernel, penalty= penalty, loss=loss, class_weight=class_weight ))]

    pipeline = Pipeline(steps)

    # create different alphas to test
    stepsize = (aHigh - aLow) / numAlphas
    alphas = np.arange(aLow, aHigh, stepsize)

    # create different l1Ratios to test
    stepsize = 1 / numL1Ratios
    l1_ratios = np.arange(l1RatLow, l1RatHigh, stepsize)

    param_grid = {'linearSVM__alpha': alphas,
                  'linearSVM__l1_ratio': l1_ratios}

    # Create the GridSearchCV
    gm_cv = GridSearchCV(pipeline, param_grid, cv=folds)

    # fit the Grid Search Cross Value Model
    gm_cv.fit(X_train, y_train)

    # Compute and print metrics
    print("Linear SVM Accuracy: {}\n".format(gm_cv.score(X_test, y_test)))
    print("Best Alpha (Learning Rate): " + str(gm_cv.best_params_.get('linearSVM__alpha')))
    print("Best L1 Ratio : " + str(gm_cv.best_params_.get('linearSVM__l1_ratio')))


def performNoNLinearSVM(X, y, impStrategy= 'mean', preprocess=StandardScaler(), folds=5, penalty='l2', loss='hinge', aLow = 0.0001, aHigh=.1, numAlphas=10,
                     class_weight=None, l1RatLow=0, l1RatHigh=.5, numL1Ratios=10):


    # use hold out validation for analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    # create pipeline for Model testing/training
    #penalty, loss, and class weights
    steps = [('imputation', Imputer(missing_values='NaN', strategy=impStrategy, axis=0)),
             ('scaler', preprocess),
             ('linearSVM', SGDClassifier(penalty= penalty, loss=loss, class_weight=class_weight ))]

    pipeline = Pipeline(steps)

    # create different alphas to test
    stepsize = (aHigh - aLow) / numAlphas
    alphas = np.arange(aLow, aHigh, stepsize)

    # create different l1Ratios to test
    stepsize = 1 / numL1Ratios
    l1_ratios = np.arange(l1RatLow, l1RatHigh, stepsize)

    param_grid = {'linearSVM__alpha': alphas,
                  'linearSVM__l1_ratio': l1_ratios}

    # Create the GridSearchCV
    gm_cv = GridSearchCV(pipeline, param_grid, cv=folds)

    # fit the Grid Search Cross Value Model
    gm_cv.fit(X_train, y_train)

    # Compute and print metrics
    print("Linear SVM Accuracy: {}\n".format(gm_cv.score(X_test, y_test)))
    print("Best Alpha (Learning Rate): " + str(gm_cv.best_params_.get('linearSVM__alpha')))
    print("Best L1 Ratio : " + str(gm_cv.best_params_.get('linearSVM__l1_ratio')))




def performNoNLinearSVM(X, y, impStrategy= 'mean', preprocess=StandardScaler(), folds=5, penalty='l2', loss='hinge', aLow = 0.0001, aHigh=.1, numAlphas=10,
                     class_weight=None, l1RatLow=0, l1RatHigh=.5, numL1Ratios=10):


    # use hold out validation for analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    # create pipeline for Model testing/training
    #penalty, loss, and class weights
    steps = [('imputation', Imputer(missing_values='NaN', strategy=impStrategy, axis=0)),
             ('scaler', preprocess),
             ('linearSVM', SGDClassifier(penalty= penalty, loss=loss, class_weight=class_weight ))]

    pipeline = Pipeline(steps)

    # create different alphas to test
    stepsize = (aHigh - aLow) / numAlphas
    alphas = np.arange(aLow, aHigh, stepsize)

    # create different l1Ratios to test
    stepsize = 1 / numL1Ratios
    l1_ratios = np.arange(l1RatLow, l1RatHigh, stepsize)

    param_grid = {'linearSVM__alpha': alphas,
                  'linearSVM__l1_ratio': l1_ratios}

    # Create the GridSearchCV
    gm_cv = GridSearchCV(pipeline, param_grid, cv=folds)

    # fit the Grid Search Cross Value Model
    gm_cv.fit(X_train, y_train)

    # Compute and print metrics
    print("Linear SVM Accuracy: {}\n".format(gm_cv.score(X_test, y_test)))
    print("Best Alpha (Learning Rate): " + str(gm_cv.best_params_.get('linearSVM__alpha')))
    print("Best L1 Ratio : " + str(gm_cv.best_params_.get('linearSVM__l1_ratio')))




X, y= getWineData()

performLinearSVM(X, y, folds=5, aHigh= .1, numL1Ratios=100)