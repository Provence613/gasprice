from django.shortcuts import render
from django.shortcuts import HttpResponse
from predict.models import Transinfo
from predict.models import Predictres
from predict.models import User
from django.forms.models import model_to_dict
from django.core import serializers
from django.http import HttpResponse, JsonResponse
from django.db.models import Count
from django.http import HttpResponseRedirect
#导入`connection`
from django.db import connection
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,BaggingRegressor
from sklearn import tree
import xgboost as xgb
import pandas as pd
import numpy as np
import os
from etherscan.proxies import Proxies
from etherscan.stats import Stats
import requests
import  urllib
from urllib.request import urlretrieve
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
import pickle
import keras
# 全局变量
with open('./api_key.json', mode='r') as key_file:
    key = json.loads(key_file.read())['key']
api = Proxies(api_key=key)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler1 = MinMaxScaler(feature_range=(0, 1))
# 导入该模块
# Create your views here.
# request参数必须有，名字是类似self的默认规则，可以改，它封装了用户请求的所有内容
def get_recent_block():
    block = api.get_most_recent_block()
    return int(block, 16)
def creat_block_list(blockheight,num):
    block_list=[]
    for i in range(num):
        block_list.append(blockheight-i)
    return block_list

def get_tran_Data(block_list):
    # height_list=[] #blockheight
    num_list=[] #transaction num
    lowprice_list=[] #low price
    maxprice_list=[]
    aveprice_list=[] #average price
    id_list=[]
    i=1
    for blockheight in block_list:
        block = api.get_block_by_number(blockheight)
        # print(type(block))
        # print("Transaction num is {}".format(len(block['transactions'])))
        num_list.append(len(block['transactions']))
        id_list.append(i)
        if len(block['transactions'])==0:
            lowprice_list.append(0)
            maxprice_list.append(0)
            aveprice_list.append(0)
        else:
            gasprice=[]
            for row in block['transactions']:
                gasprice.append(round(int(row['gasPrice'],16)/1000000000,2))
            # print("The min gasprice is {} in block {}".format(min(gasprice),blockheight))
            lowprice_list.append(min(gasprice))
            maxprice_list.append(max(gasprice))
            aveprice_list.append(round(np.mean(gasprice),2))
        i=i+1
        # height_list.append(blockheight)
        # print(int(block['transactions'][1]['gasPrice'],16))
    return id_list,num_list,lowprice_list,aveprice_list,maxprice_list
def get_tran_info(TX_HASH):
    transaction = api.get_transaction_by_hash(tx_hash=TX_HASH)
    info=[]
    info.append(transaction['hash'])
    info.append(int(transaction['blockNumber'],16))
    info.append(transaction['from'])
    info.append(transaction['to'])
    info.append(int(transaction['gas'],16))
    info.append(int(transaction['gasPrice'],16)/1000000000)
    info.append(int(transaction['value'],16))
    return info
def get_block_info(blocknumber):
    block = api.get_block_by_number(blocknumber)
    info=[]
    info.append(block['hash'])
    info.append(int(block['number'],16))
    info.append(int(block['difficulty'],16))
    info.append(int(block['timestamp'],16))
    info.append(len(block['transactions']))
    tx_hash = []
    for row in block['transactions']:
        tx_hash.append(row['hash'])
    info.append(tx_hash)
    info.append(int(block['nonce'],16))
    info.append(int(block['gasLimit'],16))
    info.append(int(block['gasUsed'],16))
    return info
def get_pre_Data(block_list):
    # height_list=[] #blockheight
    maxprice_list=[]
    lowprice_list=[] #low price
    aveprice_list=[] #average price
    timestamp_list=[]
    for blockheight in block_list:
        block = api.get_block_by_number(blockheight)
        # print(type(block))
        # print("Transaction num is {}".format(len(block['transactions'])))
        if len(block['transactions'])==0:
            lowprice_list.append(0)
            aveprice_list.append(0)
            maxprice_list.append(0)
        else:
            gasprice=[]
            for row in block['transactions']:
                gasprice.append(round(int(row['gasPrice'],16)/1000000000,2))
            # print("The min gasprice is {} in block {}".format(min(gasprice),blockheight))
            lowprice_list.append(min(gasprice))
            maxprice_list.append(max(gasprice))
            aveprice_list.append(round(np.mean(gasprice),2))
        timestamp_list.append(int(block['timestamp'],16))
        # height_list.append(blockheight)
        # print(int(block['transactions'][1]['gasPrice'],16))
    return lowprice_list,aveprice_list,maxprice_list,timestamp_list
def getrate():
    api1 = Stats(api_key=key)
    last_price = api1.get_ether_last_price()
    return last_price['ethusd']
def process_eval(start,stop):
    dataset = pd.read_csv('./transactioninfo2.csv', header=0, index_col=0)
    # test set
    values = dataset.values
    values = values.astype('float32')
    test_set = dataset[(dataset.date > start) & (dataset.date < stop)]
    # train set
    cols = ['height', 'date', 'difficulty', 'gaslimit', 'rate',  'gasprice']
    X = dataset[cols]
    y = dataset['confirmtime']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    n_train_size = X_train.shape[0]
    train_set = pd.concat([X_train, y_train], axis=1, ignore_index=False)
    all = pd.concat([train_set, test_set], axis=0, ignore_index=False)
    values = all.values
    values = values.astype('float32')
    values = values[:, 2:]
    scaledData = scaler1.fit_transform(values)
    train = scaledData[:n_train_size, :]
    test = scaledData[n_train_size:, :]
    # split into input and outputs
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]
    return  X_train, X_test, y_train, y_test
def backtest(request):
    if request.method == "POST":
        count1=int(request.POST.get("count",None))
        start=int(request.POST.get("time",None))
        type=request.POST.get("type",None)
        stop=start+20000
        flag=1
    else:
        count1=20
        type="1"
        start=26100000
        stop=26120000
        flag=0
    X_train, X_test, y_train, y_test = process_eval(start,stop)
    if type=="7":
        keras.backend.clear_session()
        model_lstm = load_model('./lstm_model_time.h5')
        test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        # print(test)
        pre = model_lstm.predict(test)
        x_test = test.reshape((test.shape[0], test.shape[2]))
        y_predict_test = pre.reshape(-1, 1)
        inv_yhat = np.concatenate((x_test, y_predict_test), axis=1)
        inv_yhat = scaler1.inverse_transform(inv_yhat)
        y_predict_test = inv_yhat[:, -1]
        # invert scaling for actual
        y_test_invert = y_test.reshape(-1, 1)
        inv_y = concatenate((x_test, y_test_invert), axis=1)
        inv_y = scaler1.inverse_transform(inv_y)
        y_test_res = inv_y[:, -1]
        df_real_predict_test = pd.DataFrame({'real': y_test_res, 'predict': y_predict_test, 'error': abs(y_predict_test - y_test_res)})
    else:
        if type == "1":
            # model = xgb.XGBRegressor(learning_rate=0.01, n_estimators=600, max_depth=7, min_child_weight=1)
            model_xgb = xgb.Booster(model_file='./xgb_time.model')
            x_test = xgb.DMatrix(X_test)
            y_predict_test = model_xgb.predict(x_test)
        elif type == "2":
            # model = RandomForestRegressor(n_estimators=550)
            # model.fit(X_train, y_train)
            with open("./model_rfr_time.pkl", "rb") as f:
                model= pickle.load(f)
            y_predict_test = model.predict(X_test)
        elif type == "3":
            # model = ExtraTreesRegressor(n_estimators=100)
            # model.fit(X_train, y_train)
            with open("./model_etr_time.pkl", "rb") as f:
                model = pickle.load(f)
            y_predict_test = model.predict(X_test)
        elif type == "4":
            # model = GradientBoostingRegressor(n_estimators=550)
            # model.fit(X_train, y_train)
            with open("./model_gbr_time.pkl", "rb") as f:
                model = pickle.load(f)
            y_predict_test = model.predict(X_test)
        elif type == "5":
            # model = LinearRegression()
            # model.fit(X_train, y_train)
            with open("./model_linear_time.pkl", "rb") as f:
                model = pickle.load(f)
            y_predict_test = model.predict(X_test)
        elif type == "6":
            # model = BaggingRegressor(tree.DecisionTreeRegressor(), n_estimators=100, max_samples=0.3)
            # model.fit(X_train, y_train)
            with open("./model_bag_time.pkl", "rb") as f:
                model= pickle.load(f)
            y_predict_test = model.predict(X_test)
        # model.fit(X_train, y_train)
        # # 测试集
        # y_predict_test = model.predict(X_test)
        X_test_invert = X_test.reshape((X_test.shape[0], X_test.shape[1]))
        y_predict_test = y_predict_test.reshape(-1, 1)
        inv_yhat = concatenate((X_test_invert[:, 0:], y_predict_test), axis=1)
        inv_yhat = scaler1.inverse_transform(inv_yhat)
        y_predict_test = inv_yhat[:, -1]
        # invert scaling for actual
        y_test_invert = y_test.reshape(-1, 1)
        inv_y = concatenate((X_test_invert, y_test_invert), axis=1)
        inv_y = scaler1.inverse_transform(inv_y)
        y_test_res = inv_y[:, -1]
        df_real_predict_test = pd.DataFrame({'real': y_test_res, 'predict': y_predict_test, 'error': abs(y_predict_test - y_test_res)})
    part_data = df_real_predict_test.head(count1)
    real_list = part_data['real'].tolist()
    pred_list = part_data['predict'].tolist()
    error_list=part_data['error'].tolist()
    df_1 = df_real_predict_test[df_real_predict_test.error < 10]
    df_2 = df_real_predict_test[(df_real_predict_test.error > 10) & (df_real_predict_test.error < 20)]
    df_3 = df_real_predict_test[(df_real_predict_test.error >20) & (df_real_predict_test.error < 30)]
    df_4 = df_real_predict_test[(df_real_predict_test.error >30) & (df_real_predict_test.error <40)]
    df_5 = df_real_predict_test[df_real_predict_test.error > 40]
    count_list = [df_1.shape[0], df_2.shape[0], df_3.shape[0], df_4.shape[0], df_5.shape[0]]
    id_list=[]
    for id in range(count1):
        id_list.append(id+1)
    label_list = ['<10', '10<20', '20<30', '30<40', '>40']
    return render(request,'clara/backtest.html',{"flag":flag,"type":type,"time":start,"count":count1,"label":json.dumps(id_list),"List1":json.dumps(real_list),"List2":json.dumps(pred_list),"List3":json.dumps(error_list),"label1":label_list,"List4":count_list})
def process_data():
    dataset = pd.read_csv('./transactioninfo3.csv', header=0, index_col=0)
    values = dataset.values
    values = values.astype('float32')
    scaledData = scaler.fit_transform(values)
    X = scaledData[:, :-1]
    y = scaledData[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
def process_test(test):
    min = [[2.65068, 21000, 76.43, 1.0]]
    max = [[2.8008, 7607448, 155.65, 200.0]]
    max = np.array(max)
    min = np.array(min)
    test = np.array(test)
    test = (test - min) / (max - min)
    # test = np.array(test)
    return test


def predict_data(test,type):
    X_train, X_test, y_train, y_test=process_data()
    if type=="1":
        model_xgb = xgb.Booster(model_file='./xgb_time.model')
        test = process_test(test)
        test = np.array(test)
        x_test = xgb.DMatrix(test)
        pre = model_xgb.predict(x_test)
        x_test = test.reshape((test.shape[0], test.shape[1]))
        y_predict_test = pre.reshape(-1, 1)
        inv_yhat = np.concatenate((x_test, y_predict_test), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        y_pred = inv_yhat[:, -1]
    elif type=="2":
        # model_rfr = RandomForestRegressor(n_estimators=550)
        # model_rfr.fit(X_train,y_train)
        with open("./model_rfr_time.pkl", "rb") as f:
            model_rfr = pickle.load(f)
        test = process_test(test)
        test = np.array(test)
        pre = model_rfr.predict(test)
        x_test = test.reshape((test.shape[0], test.shape[1]))
        y_predict_test = pre.reshape(-1, 1)
        inv_yhat = np.concatenate((x_test, y_predict_test), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        y_pred = inv_yhat[:, -1]
        # y_pred=model_rfr.predict(test)
    elif type=="3":
        # model_etr = ExtraTreesRegressor(n_estimators=100)
        # model_etr.fit(X_train,y_train)
        with open("./model_etr_time.pkl", "rb") as f:
            model_etr = pickle.load(f)
        test = process_test(test)
        test = np.array(test)
        pre = model_etr.predict(test)
        x_test = test.reshape((test.shape[0], test.shape[1]))
        y_predict_test = pre.reshape(-1, 1)
        inv_yhat = np.concatenate((x_test, y_predict_test), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        y_pred = inv_yhat[:, -1]
        # y_pred=model_etr.predict(test)
    elif type=="4":
        # model_gbr=GradientBoostingRegressor(n_estimators=550)
        # model_gbr.fit(X_train,y_train)
        with open("./model_gbr_time.pkl", "rb") as f:
            model_gbr = pickle.load(f)
        test = process_test(test)
        test = np.array(test)
        pre = model_gbr.predict(test)
        x_test = test.reshape((test.shape[0], test.shape[1]))
        y_predict_test = pre.reshape(-1, 1)
        inv_yhat = np.concatenate((x_test, y_predict_test), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        y_pred = inv_yhat[:, -1]
        # y_pred=model_gbr.predict(test)
    elif type=="5":
        # model_liner = LinearRegression()
        # model_liner.fit(X_train,y_train)
        with open("./model_linear_time.pkl", "rb") as f:
            model_liner = pickle.load(f)
        test = process_test(test)
        test = np.array(test)
        pre = model_liner.predict(test)
        x_test = test.reshape((test.shape[0], test.shape[1]))
        y_predict_test = pre.reshape(-1, 1)
        inv_yhat = np.concatenate((x_test, y_predict_test), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        y_pred = inv_yhat[:, -1]
        # y_pred=model_liner.predict(test)
    elif type=="6":
        # model_bag=BaggingRegressor(tree.DecisionTreeRegressor(), n_estimators=100, max_samples=0.3)
        # model_bag.fit(X_train,y_train)
        with open("./model_bag_time.pkl", "rb") as f:
            model_bag = pickle.load(f)
        test = process_test(test)
        test = np.array(test)
        pre = model_bag.predict(test)
        x_test = test.reshape((test.shape[0], test.shape[1]))
        y_predict_test = pre.reshape(-1, 1)
        inv_yhat = np.concatenate((x_test, y_predict_test), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        y_pred = inv_yhat[:, -1]
        # y_pred=model_bag.predict(test)
    elif type=="7":
        keras.backend.clear_session()
        model_lstm = load_model('./lstm_model_time.h5')
        test = process_test(test)
        test=[test]
        test = np.array(test)
        # print(test)
        pre = model_lstm.predict(test)
        x_test = test.reshape((test.shape[0], test.shape[2]))
        y_predict_test = pre.reshape(-1, 1)
        inv_yhat = np.concatenate((x_test, y_predict_test), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        y_pred = inv_yhat[:, -1]
    return (y_pred[0])


def forecast(request):
    type=''
    # contime=20
    time=0
    blockheight = get_recent_block()
    block = api.get_block_by_number(blockheight)
    difficulty=int(block['difficulty'],16)
    if request.method == "POST":
        gasprice=int(request.POST.get("gasprice",None))
        gaslimit=int(request.POST.get("gaslimit",None))
        type=request.POST.get("type",None)
        ethusd=getrate()
        ethusd=float(ethusd)
        test = np.array([[difficulty, gaslimit,ethusd, gasprice]])
        time=predict_data(test,type)
        time=int(time)
        if time<0 :
            time=10
    return render(request,'clara/forecast.html',{"height":blockheight,"confirmtime":time,"type":type})
# def forecast(request):
#     return render(request,"clara/forecast.html")