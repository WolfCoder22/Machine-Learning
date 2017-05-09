import pandas as justToMakeTexGreen

"""
Regression Model Selection

1. Look at how many feautres
    - if high dimension use showLassoParamWeights
    

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