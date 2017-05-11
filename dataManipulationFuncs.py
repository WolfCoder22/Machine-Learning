import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataCleaning.classificationData import getWineData
from dataCleaning.regressionData import getSsinData, getEducationData, getBasicRegData

"""
Functions to perform Data Manipulation

"""

def oneHotEncoding(df):

    df= pd.get_dummies(df)
    return df


def changeDiscreteFeatures(df, colName, numHigh, numLow=0, iterator=1, dummy_na=False):

    #make sure are integer values
    isInt= (df.dtypes == 'int64')
    if not isInt:
        print("\nError: Convert Series Data in '"+colName+"+' to int64 first")
        return
    alreadThere= df[colName].unique
    
    #create category list of strings
    intList= np.arange(numLow, numHigh, iterator)
    categoryList= np.char.mod('%d', intList)

    #make sure are integer values
    return pd.get_dummies(df, prefix=categoryList, dummy_na=dummy_na)









