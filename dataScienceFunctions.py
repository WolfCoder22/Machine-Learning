import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataCleaning.classificationData import getWineData
from dataCleaning.regressionData import getSsinData, getEducationData, getBasicRegData

global figSize
figsize=[8, 8]

"""
Functions to Analyze Data
    - change figure size with Global Var above
    
    
Visual EDA:
        
    scatterMatrixPlot(isCategorical, dfX, dfY=None, diagonal='kde')
        -plot scatter matrix
        - if categorical maek iscategorical as true and enter DfY for scatter coloring
        -diagnoal
            -'kde'- density plot of vars
            -'hist'- histogram of vars
    
Statistical EDA Data Analysis:

    getOutlierIndexBool(points, stdThresh=3.5, removeOutliers=False):
        -checks outliers in a data set with threshold with MAD criterion
        -return Boolean array of indexes
        -code from StackOverflow
        -Set remove outliers to true to remove them
        
    printClassImabalance(dfY)
        -for categorical Data only
        
"""

def scatterMatrixPlot(isCategorical, dfX, dfY=None, diagonal='kde'):

    #color data points of label if Categorical Data
    if isCategorical:
        pd.scatter_matrix(dfX, c=dfY, s=150, figsize=figsize, marker='x', diagonal=diagonal)
    else:
        pd.scatter_matrix(dfX, figsize=figsize, s=150, marker='x', diagonal=diagonal)

    #maximize window and show
    plt.show()


def getOutlierIndexBool(points, stdThresh=3.5, removeOutliers=False):

    if len(points.shape) == 1:
        points = dfX[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    mask= modified_z_score > stdThresh

    # print and return findings
    numOutliers = np.sum(mask)
    print("\nNum of points with Std Thresh" + str(stdThresh) + ": " + str(np.sum(mask)))

    if removeOutliers:

        return points[~mask]


def printClassImabalance(dfY):

    classes= dfY.unique()

    numExamples = dfY.shape
    counts=dfY.value_counts()
    percentages= ((counts/numExamples)* 10).round(2)

    numPecentdf= counts['percent']= percentages

    #print findings
    numLables= counts.shape[0]

    print("Label Counts are")
    for index in range(0, numLables-1):
        count= counts.iloc[index]
        percent = percentages.iloc[index]

        print("Label "+str(classes[index])+": count= "+str(count)+", "+str(percent)+"% of data")




dfX, dfY= getEducationData(True)
#scatterMatrixPlot(False, dfX.iloc[:, [1, 2, 3, 4 ,5, 6]], dfY)

#print(df.head())
printClassImabalance(dfY)



