import joblib
import requests

import pandas as pd
from train import (
    f_1,
    f_2
)

if __name__ == '__main__':
    url="https://drive.google.com/file/d/18vrkoFro4YfK5qSIDYbTIEhaylj2nyta/view?usp=sharing"
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    df_0 = pd.read_csv(url, index_col='Время отбора', parse_dates=['Время отбора'])
    df, features, tg = f_1(df_0)
    X_train_scaled, y_train, X_test_scaled, y_test, scaler = f_2(df, tg)


    # X_test_scaled = joblib.load("./X_test_scaled.joblib")
    # y_test = joblib.load("./y_test.joblib")

    input = {"X": X_test_scaled.tolist()}
    # input = {"X": X_test_scaled.tolist()}

    resp = requests.post(
            "http://127.0.0.1:80/predict",
            json=input
        )
    print(f"Input data: {input}")
    print(f"Predicted: {resp.json()}")
    print(f"Expected: {y_test}")
    print("----")
