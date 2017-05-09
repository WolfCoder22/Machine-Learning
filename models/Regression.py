from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import Imputer, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

"""
File contains thre different types of Linear Models
-Lass0 (L1 Penalty), Ridge(L2 Penanlty), and ElasticNet(L2, L1 Combo)

Methods
    
    performRidgeReg()

"""

def performRidgeReg(X, y, folds=5):

    #use hold out validation for analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    #create pipeline for Model testing/training
    steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
             ('scaler', StandardScaler()),
             ('ridgeReg', Ridge())]

    pipeline = Pipeline(steps)

    #create different alpha paramaters to test
    alphas = np.random.uniform(low=0, high=1, size=(50,))

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


def cleanSsinData():
    df = pd.read_csv('../testData/ssin.csv', delimiter=';')

    # get x and y
    X = df.drop('y', 1)
    y = df.y

    #fix y col
    df.y=df.y.apply(lambda x: x.split(',')[0])
    df.y= pd.to_numeric(df.y)

    return X, y

def cleanEducationData():
    df= pd.read_csv('../testData/educationData.csv', index_col=1)

    df= df.drop('deleleThis', 1)    #delete unessacry inde column

    #change categorical data
    df.Region= df.Region.astype('category')
    #print(df.Region.unique()) #check proper categorical

    df= pd.get_dummies(df)

    #get x and y
    X= df.drop('Y', 1)
    y= df.Y

    return  X, y

X, y= cleanSsinData()

#performRidgeReg(X, y, )

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
