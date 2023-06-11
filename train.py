import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectPercentile,
    mutual_info_regression,
)
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    TimeSeriesSplit,
)
from sklearn.metrics import (
    mean_squared_error,
    median_absolute_error, 
    mean_absolute_percentage_error, 
    r2_score,
)
RAND_STATE = 2007
TEST_SIZE = 0.25

def f_relative_error(y: np.ndarray, y_pred: np.ndarray, mode=0):
    error = []
    for i, j in zip(y, y_pred):
        error.append(float((j-i)/i))
    if mode == 0: return(np.mean(error)+2*np.std(error, ddof=1))
    else: return(error)

def MDAPE(actual,predicted,sample_weight=None): # Median Absolute Percentage Error
    return(np.median((np.abs(np.subtract(actual, predicted)/ actual))) * 1)

def f_MRE_0(y: np.ndarray, y_pred: np.ndarray, mode=0): #CO
    error = []
    for i, j in zip(y, y_pred):
        if (0<=i<=20) & (0<=j<=20): base = 20 
        
        if (0<=i<=75) & (0<=j<=75): base = 75 

        elif (0<=i<=500) & (0<=j<=100): base = 100
        elif (0<=i<=500) & (100<j<=500): base = i
                
        elif (0<=i<=1000) & (0<j<=500): base = 500
        elif (0<=i<=1000) & (500<j<=1000): base = i
                
        else:  base = i   
        error.append(float((j-i)/base))
    if mode == 0: return(np.mean(error)+2*np.std(error, ddof=1))
    else: return(error)

def f_metrics(y_train: pd.DataFrame, y_train_pred: pd.DataFrame, y_test: pd.DataFrame, y_test_pred: pd.DataFrame, MRE) -> pd.DataFrame:
    MAE_train, MAE_test = median_absolute_error(y_train, y_train_pred), median_absolute_error(y_test, y_test_pred) # mae
    RMSE_train, RMSE_test = mean_squared_error(y_train, y_train_pred, squared=False), mean_squared_error(y_test, y_test_pred,  squared=False) #rmse
    MAPE_train, MAPE_test = mean_absolute_percentage_error(y_train, y_train_pred), mean_absolute_percentage_error(y_test, y_test_pred) # MAPE
    MDAPE_train, MDAPE_test = MDAPE(y_train, y_train_pred), MDAPE(y_test, y_test_pred)
    R2_train, R2_test = r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)
    Median_reduced_error_train, Median_reduced_error_test = MRE(y_train, y_train_pred), MRE(y_test, y_test_pred)
    relative_error_train, relative_error_test = f_relative_error(y_train, y_train_pred), f_relative_error(y_test, y_test_pred)
    return(pd.DataFrame({'MAE':[MAE_train, MAE_test], 'RMSE':[RMSE_train,RMSE_test], 'MAPE':[MAPE_train, MAPE_test], 'MDAPE':[MDAPE_train, MDAPE_test], 
                         'R2': [R2_train, R2_test], 'MRE': [Median_reduced_error_train, Median_reduced_error_test], 'Relative_error':[relative_error_train, relative_error_test]}, index=['Train','Test']))

def f_1(df_0: pd.DataFrame
       ) -> tuple[pd.DataFrame, list, str]:
    features, tg = df_0.columns[1:], df_0.columns[0]
    tg_series = df_0[tg]
    selector_1 = SelectPercentile(mutual_info_regression, percentile=95).fit(df_0[features], df_0[tg])
    df = pd.DataFrame(selector_1.transform(df_0[features]), columns=selector_1.get_feature_names_out(features), index=df_0.index)
    df[tg] = tg_series
    return(df, features, tg)

def f_2(df: pd.DataFrame, 
        tg: str, 
        random_state: int = RAND_STATE, 
        test_size: float = TEST_SIZE
       ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, sklearn.preprocessing._data]:
    df_train, df_test = train_test_split(df, shuffle=False, random_state=RAND_STATE, test_size=TEST_SIZE)
    X_train, X_test, y_train, y_test = df_train.drop([tg], axis=1), df_test.drop([tg], axis=1), df_train[[tg]], df_test[[tg]]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return(X_train_scaled, y_train, X_test_scaled, y_test, scaler)

def f_3(X_train_scaled: np.ndarray, 
        y_train: pd.DataFrame, 
        X_test_scaled: np.ndarray, 
        y_test: pd.DataFrame
       ) -> tuple[sklearn.model_selection._search, pd.DataFrame]:
    """
    Train the model
    """
    param_grid = { 
    'n_components':  [
                      int(X_train_scaled.shape[1]*0.1),
                      int(X_train_scaled.shape[1]*0.2),
                      int(X_train_scaled.shape[1]*0.5),
                      int(X_train_scaled.shape[1]*0.8),
                      int(X_train_scaled.shape[1]*0.9), 
                      X_train_scaled.shape[1]-1]
                }
    model = PLSRegression(scale=False, max_iter=1000, tol=1e-06) 
    grid = GridSearchCV(model, param_grid=param_grid, verbose=1, 
                    cv=TimeSeriesSplit(3), 
                    n_jobs=-1,
                    scoring='neg_mean_squared_error'
                   )
    grid.fit(X_train_scaled, y_train.values)
    y_train_pred, y_test_pred = pd.DataFrame(grid.predict(X_train_scaled), index=y_train.index, columns=y_test.columns), pd.DataFrame(grid.predict(X_test_scaled), index=y_test.index, columns=y_test.columns)
    y_test_pred[y_test_pred<0], y_train_pred[y_train_pred<0] = 0, 0
    metrics = f_metrics(y_train.values, y_train_pred.values, y_test.values, y_test_pred.values, f_MRE_0)
    return(grid, metrics)

if __name__ == '__main__':
    # Train multi class classification model and save it to the working directory
    url="https://drive.google.com/file/d/18vrkoFro4YfK5qSIDYbTIEhaylj2nyta/view?usp=sharing"
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    df_0 = pd.read_csv(url, index_col='Время отбора', parse_dates=['Время отбора'])

    df, features, tg = f_1(df_0)
    X_train_scaled, y_train, X_test_scaled, y_test, scaler = f_2(df, tg)
    grid, metrics = f_3(X_train_scaled, y_train, X_test_scaled, y_test)
    # joblib.dump(X_test_scaled.tolist(), "./X_test_scaled.joblib")
    # joblib.dump(y_test.values.tolist(), "./y_test.joblib")
    joblib.dump(grid, "./model.joblib")
    print(metrics)
    # np.save('./X_test_scaled.npy', X_test_scaled) # save
    # np.save('./y_test.npy', y_test) # save
