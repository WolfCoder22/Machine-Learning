import pandas as pd
from numpy import NaN
from numpy import NaN



"""
My Scripted Functions
"""

from regressionFunctions import performRidgeReg, performLassoReg, performElasticReg
from regressionFunctions import testFitvsNumParms, showLassoParamWeights, getNewXfromLassoWeightThresh

from EDAFuncs import scatterMatrixPlot, correlation_matrix
from EDAFuncs import outliers, printClassImbalance, skewness

from dataManipulationFuncs import oneHotEncoding, concatDFHorizantaly, enocodeDiscreteDatWithinUnKnowns


def cleanTXdemo2015():

    dfDD2015= pd.read_csv('interviewData/DistrictDisparities2015.csv')
    dfDD2016 = pd.read_csv('interviewData/DistrictDisparities2016.csv')
    dfTXdemo2015 = pd.read_csv('interviewData/TXdemo2015.csv')
    dfTXdemo2016 = pd.read_csv('interviewData/TXdemo2016.csv')

    #print(dfDD2015.head()) #hasBlack African COllumn
    #print(dfTXdemo2015.head()) #hasBlack African COllumn

    dfDD2015['YEAR']= 2015
    dfDD2016['YEAR'] = 2016
    dfTXdemo2015['YEAR']= 2015
    dfTXdemo2016['YEAR']= 2016

    df1 = pd.concat([dfDD2015, dfDD2016], axis=0)
    #print(df1.tail())

    df2 = pd.concat([dfTXdemo2015, dfTXdemo2016], axis=0)

    df3= pd.melt(df2, id_vars=['DISTRICT', 'DPETALLC', 'YEAR'], value_vars=['ASIAN', 'AMERICAN INDIAN OR ALASKA NAT', 'NATIVE HAWAIIAN/OTHER PACIFIC'
                                                         'WHITE','BLACK OR AFRICAN AMERICAN','HISPANIC/LATINO'], var_name="HEADING NAME", value_name="NUMPPL")

    #print(df1.head())
    print("\n")
    # print(df3.head())

    df4= pd.merge(left=df3, right= df1, how='outer')

    df5= df4[['HEADING NAME', 'NUMPPL', 'YEAR']]

    df5sum= df4.sum()
    #print(df5sum.head())

    q2Answer= df5sum['NUMPPL']/ df5sum['DPETALLC']
    print(q2Answer)









    #print(dfTXdemo2015.head())
    #count= dfTXdemo2015['BLACK OR AFRICAN AMERICAN'].sum()

    #print(count)



cleanTXdemo2015()

