import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_regression
from xgboost import plot_importance
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import os
output_path="./output"
datafile_path="./output/transaction.csv"
# datafile_path="./output/tran_norm.csv"
cols = ['difficulty', 'gaslimit','gasused', 'gaspricel1', 'gaspricel2']
def process_data(datafile_path):
    data_df = pd.read_csv(datafile_path)
    # cols=['difficulty','gaspricel1','gaspricel2']
    X=data_df[cols]
    y=data_df['gasprice']
    arr_x=np.array(X)
    arr_y=np.array(y)
    # 回归特征选择
    # F = f_regression(arr_x,arr_y)
    # print(len(F))
    # print(F)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train.to_csv(os.path.join(output_path, 'x_train.csv'))
    X_test.to_csv(os.path.join(output_path, 'x_test.csv'))
    y_train.to_csv(os.path.join(output_path, 'y_train.csv'))
    y_test.to_csv(os.path.join(output_path, 'y_test.csv'))
    arr_xtrain=np.array(X_train)
    arr_xtest=np.array(X_test)
    arr_ytrain=np.array(y_train)
    arr_ytest=np.array(y_test)
    # print(arr_xtrain)
    return arr_xtrain,arr_xtest,arr_ytrain,arr_ytest




def predict_data(test):
    X_train, X_test, y_train, y_test=process_data(datafile_path)
    model = xgb.XGBRegressor(learning_rate=0.2, n_estimators=550, max_depth=3,min_child_weight=5)
    model.fit(X_train, y_train)
    model.get_booster().save_model(os.path.join(output_path, 'xgb1.model'))
    # make predictions for test data and evaluate

    X_test=np.array([[1.951230e+15 ,8.007778e+06 ,7.989444e+06, 1.000000e+00, 1.000000e+00]])
    # print(X_test)
    y_pred = model.predict(test)
    return (y_pred[0])
if __name__ == '__main__':
    test = np.array([[1.951230e+15, 8.007778e+06, 7.989444e+06, 1.000000e+00, 1.000000e+00]])
    print(predict_data(test))

