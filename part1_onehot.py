import pandas as pd
import numpy as np
import datetime
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
import regress

dataPath = "data/diamonds.csv"

def preprocessData(path):
    data = pd.read_csv(path)
    data.drop(['ID'], axis =1, inplace = True)

    # generate one hot encoded values
    onehotCuts = pd.get_dummies(data["cut"])
    onehotColours = pd.get_dummies(data["color"])
    onehotClarity = pd.get_dummies(data["clarity"])
    targetclass = data["price"]

    # drop the one hot encoded classes
    data = data.drop("cut", axis = 1)
    data = data.drop("color", axis = 1)
    data = data.drop("clarity", axis = 1)
    data = data.drop("price", axis = 1)

    # remove 0 values
    data = data[data['x'] !=0]
    data = data[data['y'] !=0]
    data = data[data['z'] !=0]

    # standardize the remaining fields
    scaler = StandardScaler()
    scaledData = scaler.fit_transform(data)
    data = pd.DataFrame(scaledData)

    # add the one hot encoded values back to the df
    data = data.join(onehotCuts)
    data = data.join(onehotColours)
    data = data.join(onehotClarity)  
    data = data.join(targetclass)
    # data = data.fillna(data.median())

    return data

data = preprocessData(dataPath)
data.to_csv("preprocessedData.csv", index = False)


predictions, execution_time, y_test = regress.run(data)
print("Learn: execution time={t:.3f} seconds".format(t = execution_time))

# test_predictions = rf.predict(test_data)

print("on y_test")
print("Root mean squared error: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Mean squared error: ", (metrics.mean_squared_error(y_test, predictions)))
print("Mean absolute error: ", (metrics.mean_absolute_error(y_test, predictions)))
print("R2 score: ", metrics.r2_score(y_test, predictions))



# print(data.head())