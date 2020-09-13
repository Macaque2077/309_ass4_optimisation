import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics




dataPath = "data/diamonds.csv"
cuts = ["fair", "good", "very good", "ideal", "premium"]
colours = ["E", "I", "J", "H", "G", "D", "F"]
clarity = ["SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2", "SI1", "I1", "IF"]

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
# data.to_csv("preprocessedData.csv", index = False)

train, test = train_test_split(data, test_size=0.3, random_state=309, shuffle=True)

y_train = train.iloc[:,-1]
x_train = train.iloc[:,:-1] 

y_test = test.iloc[:,-1]
x_test = test.iloc[:,:-1]

rf = LinearRegression()
rf.fit(x_train, y_train)


predictions = rf.predict(x_test)

# test_predictions = rf.predict(test_data)

print("on y_test")
print("Mean squared error: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("R2 score: ", metrics.r2_score(y_test, predictions))


print(data.head())