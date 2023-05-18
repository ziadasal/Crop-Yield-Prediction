
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_validate, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import linregress
import time

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_regression_results(ax, y_test, y_pred, title, estimated_time, scores):
    slope, intercept, rvalue, pvalue, stderr = linregress(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [intercept+y_test.min()*slope, intercept+y_test.max()*slope], '--r')
    ax.scatter(y_test, y_test, alpha=0.7, label='Actual', c='green')
    ax.scatter(y_test, y_pred, alpha=0.7, label='Predicted', c='red')

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    ax.set_xlabel('Actual values in tonnes')
    ax.set_ylabel('Predictes values in tonnes')
    ax.set_title('{}\nTrained in {:.2f} Milliseconds'.format(title, estimated_time*1000))
    

def regression_analysis(data):
    y = data['Yield (hg/ha)']
    X = data.drop('Yield (hg/ha)', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lin = LinearRegression()
    dtr = DecisionTreeRegressor()
    sgd = SGDRegressor(loss='squared_error')
    gbr = GradientBoostingRegressor()
    knn = KNeighborsRegressor(n_neighbors=5)
    rfr = RandomForestRegressor()
    estimators = [('Linear Regression', lin),
                  ('Decision Tree Regression', dtr),
                  ('Stochastic Gradient Descent Regression', sgd),
                  ('Gradient Boosting Regression', gbr),
                  ('K-nearest Neighbour 5', knn),
                  ('Random Forest Regression', rfr)]
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(20, 13))
    axs = np.ravel(axs)
    for ax, (name, est) in zip(axs, estimators):
        start_time = time.time()
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        estimated_time =  time.time() - start_time
        plot_regression_results(ax, y_test, y_pred, name, estimated_time, 
                                (r'$R^2$ = {:.2f}' + '\n' + 
                                r'MAE = {:.0f}' + '\n' +
                                r'MSE = {:.0f}' + '\n' +
                                r'RMSE = {:.0f}' + '\n' +
                                r'MAX = {:.0f}' + '\n' +
                                r'MAPE = {:.2f}%')
                                .format(r2_score(y_test, y_pred),
                                    mean_absolute_error(y_test, y_pred),
                                    mean_squared_error(y_test, y_pred),
                                    mean_squared_error(y_test, y_pred, squared=False),
                                    max_error(y_test, y_pred),
                                    mean_absolute_percentage_error(y_test, y_pred)))
    
    plt.suptitle('Regressionsverfahren')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show() 

def evaluate_estimators(data):
    r2_values = []
    max_error_values = []
    neg_mean_absolute_error_values = []
    neg_mean_squared_error_values = []
    neg_root_mean_squared_error_values = []
    y = data['Yield (hg/ha)']
    X = data.drop('Yield (hg/ha)', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lin = LinearRegression()
    dtr = DecisionTreeRegressor()
    sgd = SGDRegressor(loss='squared_error')
    gbr = GradientBoostingRegressor()
    knn = KNeighborsRegressor(n_neighbors=5)
    rfr = RandomForestRegressor()

# Verwendete Regressionen
    estimators = [('Linear Regression', lin),
                ('Decision Tree Regression', dtr),
                ('Stochastic Gradient Descent Regression', sgd),
                ('Gradient Boosting Regression', gbr),
                ('K-nearest Neighbour 5', knn),
                ('Random Forest Regression', rfr)]

    for name, est in estimators:
        # Kreuzvalidierung
        score = cross_validate(est, X_train, y_train, cv=5,
                        scoring=['r2', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error'],
                        n_jobs=-1)

        # Abspeichern der Werte
        r2_values.append(score['test_r2'])
        max_error_values.append(-score['test_max_error'])
        neg_mean_absolute_error_values.append(-score['test_neg_mean_absolute_error'])
        neg_mean_squared_error_values.append(-score['test_neg_mean_squared_error'])
        neg_root_mean_squared_error_values.append(-score['test_neg_root_mean_squared_error'])

    # Plotten der Werte
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(30, 5))

    names = ['LR', 'DTR', 'SGD', 'GBR', 'KNN', 'RFR']

    axs[0].boxplot(r2_values, labels=names)
    axs[0].set_title('R2')
    axs[1].boxplot(max_error_values, labels=names)
    axs[1].set_title('MAX')
    axs[2].boxplot(neg_mean_absolute_error_values, labels=names)
    axs[2].set_title('MAE')
    axs[3].boxplot(neg_mean_squared_error_values, labels=names)
    axs[3].set_title('MSE')
    axs[4].boxplot(neg_root_mean_squared_error_values, labels=names)
    axs[4].set_title('RMSE')

    plt.suptitle('Cross-validation')
    plt.show()

    regression = 5

    r2_mean = np.mean(r2_values[regression])
    r2_std = np.std(r2_values[regression])
    max_error_mean = np.mean(max_error_values[regression])
    max_error_std = np.std(max_error_values[regression])
    mae_mean = np.mean(neg_mean_absolute_error_values[regression])
    mae_std = np.std(neg_mean_absolute_error_values[regression])
    mse_mean = np.mean(neg_mean_squared_error_values[regression])
    mse_std = np.std(neg_mean_squared_error_values[regression])
    rmse_mean = np.mean(neg_root_mean_squared_error_values[regression])
    rmse_std = np.std(neg_root_mean_squared_error_values[regression])

    print(u'R²: {:.3f} \u00B1 {:.3f}'.format(r2_mean, r2_std))
    print(u'MAX: {:,.0f} \u00B1 {:,.0f}'.format(max_error_mean, max_error_std))
    print(u'MAE: {:,.0f} \u00B1 {:,.0f}'.format(mae_mean, mae_std))
    print(u'MSE: {:,.0f} \u00B1 {:,.0f}'.format(mse_mean, mse_std))
    print(u'RMSE: {:,.0f} \u00B1 {:,.0f}'.format(rmse_mean, rmse_std))
    