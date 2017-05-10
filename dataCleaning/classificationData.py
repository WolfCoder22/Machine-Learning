import pandas as pd

def getWineData(splitXy):
    df = pd.read_csv('data/classification/wineData.csv', index_col=0)

    if splitXy:
        X = df.drop('Class', 1)
        y = df.Class
        return X, y

    else:
        return df