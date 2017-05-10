import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataCleaning.classificationData import getWineData
from dataCleaning.regressionData import getSsinData, getEducationData, getBasicRegData

global figSize
figsize=[8, 8]

"""
Functions to Analyze Data
    - change figure size with global above
    - 

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

dfX, dfY= getWineData(True)
scatterMatrixPlot(False, dfX.iloc[:, [1, 2, 3, 4 ,5, 6]], dfY)




