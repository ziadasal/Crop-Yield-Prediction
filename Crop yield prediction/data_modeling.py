import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import time

def run_regression(data, model):
    y = data['Yield (hg/ha)']
    X = data.drop('Yield (hg/ha)', axis=1)

    if (model == 'Linear Regression'):
        model = LinearRegression()
    elif (model == 'Decision Tree Regression'):
        model = DecisionTreeRegressor()
    elif (model == 'Stochastic Gradient Descent Regression'):
        model = SGDRegressor()
    elif (model == 'Gradient Boosting Regression'):
        model = GradientBoostingRegressor()
    elif (model == 'Random Forest Regression'):
        model = RandomForestRegressor()
    elif (model == 'K-nearest Neighbour 5'):
        model = KNeighborsRegressor(n_neighbors=5)
    else:
        print('Error: Model not found')
        return
    
    results = {}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    estimated_time =  time.time() - start_time

    slope, intercept, rvalue, pvalue, stderr = linregress(y_test, y_pred)

    scores = (r'$R^2$ = {:.2f}' + '\n' + 
                r'MAE = {:.0f}' + '\n' +
                r'MSE = {:.0f}' + '\n' +
                r'RMSE = {:.0f}' + '\n' +
                r'MAX = {:.0f}' + '\n' +
                r'MAPE = {:.2f}%').format(r2_score(y_test, y_pred),
                                        mean_absolute_error(y_test, y_pred),
                                        mean_squared_error(y_test, y_pred),
                                        mean_squared_error(y_test, y_pred, squared=False),
                                        max_error(y_test, y_pred),
                                        np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

    results[model] = {'y_test': y_test, 'y_pred': y_pred, 'name': model, 'scores': scores, 'slope': slope, 'intercept': intercept}

    # Create the plot
    fig, ax = plt.subplots(figsize=(9, 9))

    result = results[model]
    ax.plot([result['y_test'].min(), result['y_test'].max()], [result['intercept']+result['y_test'].min()*result['slope'], result['intercept']+result['y_test'].max()*result['slope']], '--r')
    ax.scatter(result['y_test'], result['y_test'], alpha=0.7, label='Actual', c='green')
    ax.scatter(result['y_test'], result['y_pred'], alpha=0.7, label='Predicted', c='red')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra],[scores],loc='upper left')

    ax.set_xlabel('Actual values in tonnes')
    ax.set_ylabel('Predicted values in tonnes')
    ax.set_title('{}\nTrained in {:.2f} Milliseconds'.format(model, estimated_time*1000))

    plt.tight_layout()
    plt.show()






