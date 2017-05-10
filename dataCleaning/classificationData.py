import pandas as pd

def getWineData():
    df = pd.read_csv('../../testData/wineData.csv')

    # get x and y
    X = df.drop('Class', 1)
    y = df.Class


    return X, y
