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
    
    
Plotting:
        
    scatterMatrixPlot(isCategorical, dfX, dfY=None, diagonal='kde')
        -plot scatter matrix
        - if categorical maek iscategorical as true and enter DfY for scatter coloring
        -diagnoal
            -'kde'- density plot of vars
            -'hist'- histogram of vars
    
Expolatory Data Anlaysis:

    is_outlier(points, stdThresh=3.5):
        -checks outliers in a data set with threshold
        -return Boolean array of indexes
        -code from StackOverflow
        
"""

"""


"""
def scatterMatrixPlot(isCategorical, dfX, dfY=None, diagonal='kde'):

    #color data points of label if Categorical Data
    if isCategorical:
        pd.scatter_matrix(dfX, c=dfY, s=150, figsize=figsize, marker='x', diagonal=diagonal)
    else:
        pd.scatter_matrix(dfX, figsize=figsize, s=150, marker='x', diagonal=diagonal)

    #maximize window and show
    plt.show()

def is_outlier(points, stdThresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > stdThresh

dfX, dfY= getWineData(True)
#scatterMatrixPlot(False, dfX.iloc[:, [1, 2, 3, 4 ,5, 6]], dfY)

print(is_outlier(dfX))




