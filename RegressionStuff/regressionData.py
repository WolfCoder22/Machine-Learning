import pandas as pd

def getBasicRegData():
    df = pd.read_csv('../testData/basicRegData.csv')

    # get x and y
    X = df.drop('Y', 1)
    y = df.Y

    return X, y

def getSsinData():
    df = pd.read_csv('../testData/ssin.csv', delimiter=';')

    #fix y col
    df.y=df.y.apply(lambda x: x.split(',')[0])
    df.y= pd.to_numeric(df.y)

    # get x and y
    X = df.drop('y', 1)
    y = df.y


    return X, y

def getEducationData():
    df= pd.read_csv('../testData/educationData.csv', index_col=1)

    df= df.drop('deleleThis', 1)    #delete unessacry inde column

    #change categorical data
    df.Region= df.Region.astype('category')
    #print(df.Region.unique()) #check proper categorical

    df= pd.get_dummies(df)

    #get x and y
    X= df.drop('Y', 1)
    y= df.Y

    return  X, y