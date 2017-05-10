import pandas as toMakeTextGreen

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