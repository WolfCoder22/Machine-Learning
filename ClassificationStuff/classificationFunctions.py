import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler, MaxAbsScaler

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
             ('linearSVM', SGDClassifier(penalty= penalty, loss=loss, class_weight=class_weight, random_state=2))]

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


"""
Kernel types
    -rbf
    -poly

"""

def performNoNLinearSVM(X, y, kernel='rbf', impStrategy= 'mean', preprocess=StandardScaler(), folds=5, loss='hinge', cLow=.000001, cHigh=1, numCs=10,
                        gammaLow=0, gammaHigh=1, numGammas=10, class_weight=None):


    # use hold out validation for analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    # create pipeline for Model testing/training
    #penalty, loss, and class weights
    steps = [('imputation', Imputer(missing_values='NaN', strategy=impStrategy, axis=0)),
             ('scaler', preprocess),
             ('nonLinearSVM', SVC(kernel=kernel, probability=True, class_weight=class_weight, random_state=2))]

    pipeline = Pipeline(steps)

    # create different c (penalty) values to test
    stepsize = (cHigh - cLow) / numCs
    Cs = np.arange(cLow, cHigh, stepsize)

    # create different l1Ratios to test
    stepsize = 1 / numGammas
    Gammas = np.arange(gammaLow, gammaHigh, stepsize)

    param_grid = {'nonLinearSVM__C': Cs,
                  'nonLinearSVM__gamma': Gammas}

    # Create the GridSearchCV
    gm_cv = GridSearchCV(pipeline, param_grid, cv=folds)


    # fit the Grid Search Cross Value Model
    gm_cv.fit(X_train, y_train)

    # Compute and print metrics
    print("NonLinear SVM Accuracy: {}\n".format(gm_cv.score(X_test, y_test)))
    print("Best C (Penalty Of Error term): " + str(gm_cv.best_params_.get('nonLinearSVM__C')))
    print("Best Gamma Value : " + str(gm_cv.best_params_.get('nonLinearSVM__gamma')))


def performKNN(X, y, impStrategy= 'mean', preprocess=StandardScaler(), folds=5, nLow=1, nHigh=5, nIter=1):


    # use hold out validation for analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    # create pipeline for Model testing/training
    steps = [('imputation', Imputer(missing_values='NaN', strategy=impStrategy, axis=0)),
             ('scaler', preprocess),
             ('knn', KNeighborsClassifier(random_state=2))]

    pipeline = Pipeline(steps)

    # compute KNN array
    kSizes = np.arange(nLow, nHigh, nIter)

    param_grid = {'knn__n_neighbors': kSizes}

    # Create the GridSearchCV
    gm_cv = GridSearchCV(pipeline, param_grid, cv=folds)

    # fit the Grid Search Cross Value Model
    gm_cv.fit(X_train, y_train)

    # Compute and print metrics
    print("Knn Accuracy: {}\n".format(gm_cv.score(X_test, y_test)))
    print("Best Neighbor: " + str(gm_cv.best_params_.get('knn__n_neighbors')))



def performRandomForest(X, y, folds=5, impStrategy= 'mean', class_weight=None, preprocess=StandardScaler(), treeNumLow=10, treeNumhigh=11, treeNumLowIter=1):


    # use hold out validation for analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    # create pipeline for Model testing/training
    steps = [('imputation', Imputer(missing_values='NaN', strategy=impStrategy, axis=0)),
             ('scaler', preprocess),
             ('forest', RandomForestClassifier(class_weight=class_weight))]

    pipeline = Pipeline(steps)

    # compute KNN array
    treesNum = np.arange(treeNumLow, treeNumhigh, treeNumLowIter)

    param_grid = {'forest__n_estimators': treesNum}

    # Create the GridSearchCV
    gm_cv = GridSearchCV(pipeline, param_grid, cv=folds)

    # fit the Grid Search Cross Value Model
    gm_cv.fit(X_train, y_train)

    # Compute and print metrics
    print("Random Forest Accuracy: {}\n".format(gm_cv.score(X_test, y_test)))
    print("Best tree Amount: " + str(gm_cv.best_params_.get('forest__n_estimators')))

"""
use partial fit when data set extremly big and tough to fit in memory

type Param
    GaussianNB
        -Features expected to be Guassian
    MultinomialNB
        - discrete features
        -Ex. Count of a word in a textbody
        -features can consist of something like word count
    BernoulliNB
        -discrete variables that are binary

Partial Fit
    -if extremely high amount of data
    



"""

def performGuassianNB(X, y, partialFit=False, ):
    addNext=2


def performMultiNomNB(X, y, binaryData=False, folds=5, impStrategy= 'mean', preprocess=MaxAbsScaler(), aLow=0, aHigh=1, numAlphas=10, fit_prior=False, class_prior=None):
    # use hold out validation for analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #use bernulli or Multinominal Classifier based if binary data or not
    if binaryData:
        Nbclf= BernoulliNB(fit_prior=fit_prior, class_prior=class_prior)
    else:
        Nbclf= MultinomialNB(fit_prior=fit_prior, class_prior=class_prior)

    # create pipeline for Model testing/training
    steps = [('imputation', Imputer(missing_values='NaN', strategy=impStrategy, axis=0)),
             ('scaler', preprocess),
             ('nb', Nbclf)]

    pipeline = Pipeline(steps)
    print(pipeline.get_params().keys())

    # create different alphas to test
    stepsize = (aHigh - aLow) / numAlphas
    alphas = np.arange(aLow, aHigh, stepsize)

    param_grid = {'nb__alpha': alphas}

    # Create the GridSearchCV
    gm_cv = GridSearchCV(pipeline, param_grid, cv=folds)

    # fit the Grid Search Cross Value Model
    gm_cv.fit(X_train, y_train)

    # Compute and print metrics
    print("NB Accuracy: {}\n".format(gm_cv.score(X_test, y_test)))
    print("Best Alpha (smoothing paramter): " + str(gm_cv.best_params_.get('nb__alpha')))






X, y= getWineData()

performMultiNomNB(X, y)