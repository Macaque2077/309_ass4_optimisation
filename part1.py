import pandas as pd

dataPath = "data/diamonds.csv"
cuts = ["fair", "good", "very good", "ideal", "premium"]
colours = ["E", "I", "J", "H", "G", "D", "F"]
clarity = ["SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2", "SI1", "I1"]

def preprocessData(path):
    data = pd.read_csv(path)
    data.drop(['ID'], axis =1, inplace = True)

    for i, row in enumerate(data["cut"]): 
        if data["cut"][i].lower() in cuts:
            data.set_value(i, "cut", cuts.index(row.lower()))
    for j, row in enumerate(data["color"]): 
        if data["color"][j] in colours:
            data.set_value(j, "color", colours.index(row))
    for z, row in enumerate(data["clarity"]): 
        if data["clarity"][z] in clarity:
            data.set_value(z, "clarity", clarity.index(row))
    
    data = data.fillna(data.median())
    
    return data

data = preprocessData(dataPath)
data.to_csv("preprocessedData.csv", index = False)
print(data.head())