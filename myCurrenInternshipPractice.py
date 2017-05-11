import pandas as pd

from numpy import NaN

#my functions
from dataManipulationFuncs import oneHotEncoding

"""
PRATICING ON MY CURRENT INTERNSHIP WITH NEW KNOWLEDGE LEARN THIS WEEK

Steps Taken for Practice
    1. Clean Individual CVS Seperated by Type of Feature
"""


"""
1. Clean data

"""

def cleanNAICSdata():
    df= pd.read_csv('data/myCurrInternshipData/NAICSfeats.csv', index_col=0)
    df.info()
    print(df.shape)
    print(df.head())

    print('\n')
    print(df.TwoDigitNAICS.value_counts(dropna=False))

    #found MI String in Data, so making Null
    df[df.TwoDigitNAICS=='MI']= NaN

    print(df.TwoDigitNAICS.value_counts(dropna=False))

    #Turn to Categorical
    df.TwoDigitNAICS= df.TwoDigitNAICS.astype('category')
    print('\n')
    df.info()

    #check Number of Categories
    print(df.TwoDigitNAICS.unique)

    #Do one Hot encoding Later

    # df= oneHotEncoding(df)
    # print('\n')
    # print(df.head())



def cleanSqFtdata():
    return pd.read_csv('data/myCurrInternshipData/squareFootageFeat.csv', index_col=0)

def cleanWebDomainFeat():
    df = pd.read_csv('data/myCurrInternshipData/webDomainFeat.csv', index_col=0)

    print(df.WebsiteTLD.value_counts(dropna=False))

    # found NONE String in Index, so adding it to NaN count and removing
    df[df.WebsiteTLD == 'NONE'] = NaN

    print(df.WebsiteTLD.value_counts(dropna=False))

    #fix com and COM
    df[df.WebsiteTLD == 'com'] = 'COM'
    print(df.WebsiteTLD.value_counts(dropna=False))

    # Turn to Categorical
    df.TwoDigitNAICS = df.WebsiteTLD.astype('category')
    print('\n')
    df.info()

    # Do one Hot encoding Later

    # df= oneHotEncoding(df)
    # print('\n')
    # print(df.head())

cleanWebDomainFeat()