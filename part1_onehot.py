import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler 




dataPath = "data/diamonds.csv"
# cuts = ["fair", "good", "very good", "ideal", "premium"]
# colours = ["E", "I", "J", "H", "G", "D", "F"]
# clarity = ["SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2", "SI1", "I1", "IF"]

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
    # data = data.replace(0, np.nan)
    # data = data.dropna(how='any', axis=0)
    # data = data.replace(np.nan, 0)
    data = data[data['x'] !=0]
    data = data[data['y'] !=0]
    data = data[data['z'] !=0]

    # standardize the remaining fields
    # scaler = StandardScaler()
    # scaledData = scaler.fit_transform(data)
    # data = pd.DataFrame(scaledData)

    # add the one hot encoded values back to the df
    data = data.join(onehotCuts)
    data = data.join(onehotColours)
    data = data.join(onehotClarity)  
    data = data.join(targetclass)
    # data = data.fillna(data.median())
    
    # print(data.head())
    return data

data = preprocessData(dataPath)
data.to_csv("preprocessedData.csv", index = False)

train, test = train_test_split(data, test_size=0.3, random_state=309, shuffle=True)

y_train = train.iloc[:,-1]
x_train = train.iloc[:,:-1] 

y_test = test.iloc[:,-1]
x_test = test.iloc[:,:-1]

# regressors
#  rf = LinearRegression()
rf = KNeighborsRegressor(leaf_size=100) #ball tree kd_tree, brute
start_time = datetime.datetime.now()  # Track learning starting time
rf.fit(x_train, y_train)
end_time = datetime.datetime.now()  # Track learning ending time
exection_time = (end_time - start_time).total_seconds()  # Track execution time
# Step 4: Results presentation
print("Learn: execution time={t:.3f} seconds".format(t = exection_time))


predictions = rf.predict(x_test)

# test_predictions = rf.predict(test_data)

print("on y_test")
print("Root mean squared error: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Mean squared error: ", (metrics.mean_squared_error(y_test, predictions)))
print("Mean absolute error: ", (metrics.mean_absolute_error(y_test, predictions)))
print("R2 score: ", metrics.r2_score(y_test, predictions))



# print(data.head())