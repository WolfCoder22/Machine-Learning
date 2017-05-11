import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataCleaning.classificationData import getWineData
from dataCleaning.regressionData import getSsinData, getEducationData, getBasicRegData

"""
Functions to perform Data Manipulation

Methods:
    oneHotEncoding(df)
        -performs one hot encoding on a Pandas df/series
        -changes collumns with dtype categorical

    enocodeDiscreteDatWithinUnKnowns(dfFull, df, colName, numHigh, numLow=0, iterator=1)
        -does a one hot encoding for discrete Data with unfilled Data

"""

def oneHotEncoding(df):

    df= pd.get_dummies(df)
    return df

def imputeValues(df):

    df[any
    return df


def concatDFHorizantaly(dfList):

    #concate all horizantally
    fullDf= dfList[0]
    dfList.pop(0)
    for df in dfList:
        fullDf = pd.concat([fullDf, df], axis=1, join='inner', ignore_index=True)


    return fullDf


def enocodeDiscreteDatWithinUnKnowns(dfFull, df, colName, numHigh, numLow=0, iterator=1):

    #make sure are integer values
    isInt= (df.dtypes == 'int64')
    if not isInt:
        print("\nError: Convert Series Data in '"+colName+"+' to int64 first")
        return

    #create category list of strings
    intList= np.arange(numLow, numHigh, iterator)

    #add new collumns
    numNewRows= (numHigh-numLow)
    for x, val in np.ndenumerate(intList):
        newColName= colName+"_"+str(val)

        seriesData= [0]* numNewRows

        #add postive encoding
        seriesData[val]=1
        dfFull[newColName]= pd.Series(seriesData)

    #drop original disrctor coll
    dfFull = dfFull.drop(colName, 1)

    return dfFull











