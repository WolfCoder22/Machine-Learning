import pandas as pd

"""
Various functions to clean Data simnple test Data for Regression Models

"""

def getBasicRegData(splitXy):
    df = pd.read_csv('data/regression/basicRegData.csv')

    # get x and y
    if splitXy:
        X = df.drop('Y', 1)
        y = df.Y
        return X, y
    else:
        return df

def getSsinData(splitXy):
    df = pd.read_csv('data/regression/basicRegData.csv', delimiter=';')

    #fix y col
    df.y=df.y.apply(lambda x: x.split(',')[0])
    df.y= pd.to_numeric(df.y)

    # get x and y
    if splitXy:
        X = df.drop('y', 1)
        y = df.y
    else:
        return df


    return X, y

def getEducationData(splitXy):
    df= pd.read_csv('data/regression/basicRegData.csv', index_col=1)

    df= df.drop('deleleThis', 1)    #delete unnecessary index column

    #change categorical data
    df.Region= df.Region.astype('category')
    #print(df.Region.unique()) #check proper categorical

    # get x and y
    if splitXy:
        X = df.drop('Y', 1)
        y = df.Y

        return X, y
    else:
        return pd.get_dummies(df)
