import pandas as pd
from numpy import NaN
from numpy import NaN



"""
My Scripted Functions
"""
from classificationFunctions import performGaussianNB, performMultiNomNB, performKNN, \
    performRandomForest, performLinearSVM, performNoNLinearSVM

from regressionFunctions import performRidgeReg, performLassoReg, performElasticReg
from regressionFunctions import testFitvsNumParms, showLassoParamWeights, getNewXfromLassoWeightThresh

from EDAFuncs import scatterMatrixPlot, correlation_matrix
from EDAFuncs import outliers, printClassImbalance, skewness

from dataManipulationFuncs import oneHotEncoding, enocodeDiscreteDatWithinUnKnowns



