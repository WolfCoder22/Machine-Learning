import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm

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
    
    correlation_matrix(dfx)
        -plot correlation matrix
    
Statistical EDA Data Analysis:

    outliers(points, stdThresh=3.5, removeOutliers=False):
        -checks outliers in a data set with threshold with MAD criterion
        -return Boolean array of indexes
        -code from StackOverflow
        -Set remove outliers to true to remove them
        
    printClassImbalance(dfY)
        -for categorical Data only
        
    skewness(dfX, removeBadSkew=False, absGoodSkewThresh=2)
        -prints skewness of each collumn
        -set removeBadSkew to remove collumns with skewness above or below the absGoodSkewThresh
        
"""

def scatterMatrixPlot(isCategorical, dfX, dfY=None, diagonal='kde'):

    #color data points of label if Categorical Data
    if isCategorical:
        pd.scatter_matrix(dfX, c=dfY, s=150, figsize=figsize, marker='x', diagonal=diagonal)
    else:
        pd.scatter_matrix(dfX, figsize=figsize, s=150, marker='x', diagonal=diagonal)

    #maximize window and show
    plt.show()



def outliers(points, stdThresh=3.5, removeOutliers=False):

    if len(points.shape) == 1:
        points = points[:,None]
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



def correlation_matrix(dfX):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(dfX.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)

    plt.title('Abalone Feature Correlation')
    labels=['Sex','Length','Diam','Height','Whole','Shucked','Viscera','Shell','Rings',]

    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()



def printClassImbalance(dfY):

    classes= dfY.unique()

    numExamples = dfY.shape
    counts=dfY.value_counts()
    percentages= ((counts/numExamples)* 10).round(2)

    #print findings
    numLables= counts.shape[0]

    print("Label Counts are")
    for index in range(0, numLables-1):
        count= counts.iloc[index]
        percent = percentages.iloc[index]

        print("Label "+str(classes[index])+": count= "+str(count)+", "+str(percent)+"% of data")


def skewness(dfX, removeBadSkew=False, absGoodSkewThresh=2):
    skew=dfX.skew(axis=0)

    # return new df removing collumns out of renage of good skewThesh
    if removeBadSkew:
        rows,_= dfX.shape
        criteria=skew.abs() > absGoodSkewThresh

        goodSkewDf = dfX[criteria.index[criteria]]


        return goodSkewDf

    # print skewness to show
    else:
        print(skew)



