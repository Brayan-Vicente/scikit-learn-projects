import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __name__=="__main__":
    dataset = pd.read_csv('./data/whr2017.csv')
    print(dataset.describe())

    # Renombra las columnas directamente en el DataFrame
    dataset = dataset.rename(columns={
        'Economy..GDP.per.Capita.': 'gdp',
        'Family': 'family',
        'Health..Life.Expectancy.': 'lifeexp',
        'Freedom': 'freedom',
        'Trust..Government.Corruption.': 'corruption',
        'Generosity': 'generosity',
        'Dystopia.Residual': 'dystopia',
        'Happiness.Score': 'score'
    })

    X = dataset[['gdp', 'family', 'lifeexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = dataset[['score']]

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = modelLinear.predict(X_test)

    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print('linear loss: ', linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print('lasso loss: ', lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print('Ridge loss: ', ridge_loss)

    print("="*32)
    print('Coef LASSO')
    print(modelLasso.coef_)

    print("="*32)
    print('Coef RIDGE')
    print(modelRidge.coef_)


