from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import datetime 

def run(data):
    train, test = train_test_split(data, test_size=0.3, random_state=309, shuffle=True)

    y_train = train.iloc[:,-1]
    x_train = train.iloc[:,:-1] 

    y_test = test.iloc[:,-1]
    x_test = test.iloc[:,:-1]

    # grid search method Decision tree
    # params = {'max_leaf_nodes': list(range(2,200)), 'max_features':['auto', 'sqrt','log2'] ,
    # 'min_samples_split':[2,3,4], 'splitter':["best", "random"]}
    # grid_search_cv = GridSearchCV(DecisionTreeRegressor( ), params, verbose=1, cv=3)

    # # grid search method Decision tree
    # params = {'max_leaf_nodes': list(range(2,200)), 'max_features':['auto', 'sqrt','log2'] ,
    # 'min_samples_split':[2,3,4], 'criterion':["mse", "mae"]}
    # rf = GridSearchCV(RandomForestRegressor( ), params, verbose=1, cv=3)

    # Gradient Boosting params
    # params = {'n_estimators': 1000,
    #       'max_depth': 5,
    #       'min_samples_split': 2,
    #       'learning_rate': 0.05,
    #       'loss': 'ls'}




    # regressors
    #  rf = LinearRegression()
    # rf = KNeighborsRegressor() #ball tree kd_tree, brute  
    # rf = Ridge(alpha=0.5)
    # rf = DecisionTreeRegressor(max_features='auto', max_leaf_nodes=190, min_samples_split=4, splitter='random')
    # rf = RandomForestRegressor()
    # rf = GradientBoostingRegressor(**params)
    # rf = SGDRegressor(penalty='l1', max_iter=10000, random_state=309, alpha=0.001)
    rf = SVR(kernel='linear', cache_size=400, C=0.7)

    start_time = datetime.datetime.now()  # Track learning starting time
    rf.fit(x_train, y_train)
    # grid_search_cv.fit(x_train, y_train)
    end_time = datetime.datetime.now()  # Track learning ending time
    execution_time = (end_time - start_time).total_seconds()

    # print(grid_search_cv.best_estimator_)
    predictions = rf.predict(x_test)
    # predictions = grid_search_cv.predict(x_test)
    return predictions, execution_time, y_test