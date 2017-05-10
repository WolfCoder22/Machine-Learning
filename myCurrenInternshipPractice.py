import pandas as pd

"""
PRATICING ON MY CURRENT INTERNSHIP WITH NEW KNOWLEDGE LEARN THIS WEEK

Steps Taken for Practice
    1. Clean Individual CVS Seperated by Type of Feature
"""


"""
1. Clean data

"""

def cleanNAICSdata():
    df= pd.read_csv('data/myCurrInternshipData/NAICSfeats.csv')
    print(df.head())

cleanNAICSdata()