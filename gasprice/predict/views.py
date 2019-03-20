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
# predict confirmtime
scaler2 = MinMaxScaler(feature_range=(0, 1))
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

    dataset = pd.read_csv('./blockinfo2.csv', header=0, index_col=0)
    # test set
    values = dataset.values
    values = values.astype('float32')
    test_set = dataset[(dataset.date > start) & (dataset.date < stop)]
    # train set
    cols = ['height', 'date', 'gaspricel1','difficulty', 'gaslimit', 'rate',  'confirmtime', 'transaction_count','size', 'reward']
    X = dataset[cols]
    y = dataset['gasprice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    n_train_size = X_train.shape[0]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train = y_train.reshape(-1, 1)
    # train_set = pd.concat([X_train, y_train], axis=1, ignore_index=False)
    # all = pd.concat([train_set, test_set], axis=0, ignore_index=False)
    train_set = np.concatenate((X_train, y_train), axis=1)
    # print(train_set.shape)
    # print(type(train_set))
    test_set = np.array(test_set)
    all = np.concatenate((train_set, test_set), axis=0)
    # values = all.values
    # values = values.astype('float32')
    values = all[:, 2:]
    # values = all.values
    # values = values.astype('float32')
    # values = values[:, 2:]
    scaledData = scaler1.fit_transform(values)
    train = scaledData[:n_train_size, :]
    test = scaledData[n_train_size:, :]
    # split into input and outputs
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]
    return  X_train, X_test, y_train, y_test
def eval(request):
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
    #scaler1是model evaluation
    if type=="7":
        keras.backend.clear_session()
        model_lstm = load_model('./lstm_model_price.h5')
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
            model_xgb = xgb.Booster(model_file='./xgb.model')
            x_test = xgb.DMatrix(X_test)
            y_predict_test = model_xgb.predict(x_test)
            # model = xgb.XGBRegressor(learning_rate=0.01, n_estimators=600, max_depth=7, min_child_weight=1)
            # model.fit(X_train, y_train)
            # y_predict_test = model.predict(X_test)
        elif type == "2":
            # model = RandomForestRegressor(n_estimators=550)
            # model.fit(X_train, y_train)
            with open("./model_rf.pkl", "rb") as f:
                model = pickle.load(f)
            y_predict_test = model.predict(X_test)
        elif type == "3":
            # model = ExtraTreesRegressor(n_estimators=100)
            # model.fit(X_train, y_train)
            with open("./model_etr.pkl", "rb") as f:
                model = pickle.load(f)
            y_predict_test = model.predict(X_test)
        elif type == "4":
            # model = GradientBoostingRegressor(n_estimators=550)
            # model.fit(X_train, y_train)
            with open("./model_gbr.pkl", "rb") as f:
                model= pickle.load(f)
            y_predict_test = model.predict(X_test)
        elif type == "5":
            # model = LinearRegression()
            # model.fit(X_train, y_train)
            with open("./model_linear.pkl", "rb") as f:
                model= pickle.load(f)
            y_predict_test = model.predict(X_test)
        elif type == "6":
            # model = BaggingRegressor(tree.DecisionTreeRegressor(), n_estimators=100, max_samples=0.3)
            # model.fit(X_train, y_train)
            with open("./model_bag.pkl", "rb") as f:
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
    df_1 = df_real_predict_test[df_real_predict_test.error < 1]
    df_2 = df_real_predict_test[(df_real_predict_test.error > 1) & (df_real_predict_test.error < 5)]
    df_3 = df_real_predict_test[(df_real_predict_test.error > 5) & (df_real_predict_test.error < 10)]
    df_4 = df_real_predict_test[(df_real_predict_test.error > 10) & (df_real_predict_test.error < 15)]
    df_5 = df_real_predict_test[df_real_predict_test.error > 15]
    count_list = [df_1.shape[0], df_2.shape[0], df_3.shape[0], df_4.shape[0], df_5.shape[0]]
    id_list=[]
    for id in range(count1):
        id_list.append(id+1)
    label_list = ['<1', '1<5', '5<10', '10<15', '>15']
    return render(request,'eval.html',{"flag":flag,"type":type,"time":start,"count":count1,"label":json.dumps(id_list),"List1":json.dumps(real_list),"List2":json.dumps(pred_list),"List3":json.dumps(error_list),"label1":label_list,"List4":count_list})
def modeleval(request):
    if request.method == "POST":
        count1=int(request.POST.get("count",None))
    else:
        count1=20
    cursor = connection.cursor()
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=1 group by i_p;")
    results1 = cursor.fetchall()
    label_list = ['<1', '1<5', '5<10', '10<15', '>15']
    count_list=[]
    for row in results1:
        trancount = row[1]
        count_list.append(trancount)
    real_list1 = []
    pred_list1= []
    id_list1 = []
    cursor.execute("select * from predictres where id>0 and uid=1 limit %s" % (count1))
    results11 = cursor.fetchall()
    for row in results11:
        id = row[0]
        real = row[1]
        pred = row[2]
        id_list1.append(id)
        real_list1.append(real)
        pred_list1.append(pred)
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=2 group by i_p;")
    results2 = cursor.fetchall()
    count_list2 = []
    for row in results2:
        trancount = row[1]
        count_list2.append(trancount)
    real_list2 = []
    pred_list2= []
    cursor.execute("select * from predictres where id>0 and uid=2 limit %s" % (count1))
    results22 = cursor.fetchall()
    for row in results22:
        real = row[1]
        pred = row[2]
        real_list2.append(real)
        pred_list2.append(pred)
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=3 group by i_p;")
    results3 = cursor.fetchall()
    count_list3 = []
    for row in results3:
        trancount = row[1]
        count_list3.append(trancount)
    real_list3 = []
    pred_list3 = []
    cursor.execute("select * from predictres where id>0 and uid=3 limit %s" % (count1))
    results33 = cursor.fetchall()
    for row in results33:
        real = row[1]
        pred = row[2]
        real_list3.append(real)
        pred_list3.append(pred)
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=4 group by i_p;")
    results4 = cursor.fetchall()
    count_list4 = []
    for row in results4:
        trancount = row[1]
        count_list4.append(trancount)
    real_list4= []
    pred_list4 = []
    cursor.execute("select * from predictres where id>0 and uid=4 limit %s" % (count1))
    results44 = cursor.fetchall()
    for row in results44:
        real = row[1]
        pred = row[2]
        real_list4.append(real)
        pred_list4.append(pred)
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=5 group by i_p;")
    results5 = cursor.fetchall()
    count_list5 = []
    for row in results5:
        trancount = row[1]
        count_list5.append(trancount)
    real_list5 = []
    pred_list5 = []
    cursor.execute("select * from predictres where id>0 and uid=5 limit %s" % (count1))
    results55 = cursor.fetchall()
    for row in results55:
        real = row[1]
        pred = row[2]
        real_list5.append(real)
        pred_list5.append(pred)
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=6 group by i_p;")
    results6 = cursor.fetchall()
    count_list6 = []
    for row in results6:
        trancount = row[1]
        count_list6.append(trancount)
    real_list6 = []
    pred_list6 = []
    cursor.execute("select * from predictres where id>0 and uid=6 limit %s" % (count1))
    results66 = cursor.fetchall()
    for row in results66:
        real = row[1]
        pred = row[2]
        real_list6.append(real)
        pred_list6.append(pred)
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=7 group by i_p;")
    results7 = cursor.fetchall()
    count_list7 = []
    for row in results7:
        trancount = row[1]
        count_list7.append(trancount)
    real_list7 = []
    pred_list7 = []
    cursor.execute("select * from predictres where id>0 and uid=7 limit %s" % (count1))
    results77 = cursor.fetchall()
    for row in results77:
        real = row[1]
        pred = row[2]
        real_list7.append(real)
        pred_list7.append(pred)
    return render(request,'modeleval.html',{"label1":label_list,"List1":count_list,"List2":count_list2,"List3":count_list3,"List4":count_list4,"List5":count_list5,"List6":count_list6,"List7":count_list7,"id_list1":id_list1,"real_list1":real_list1,"pred_list1":pred_list1,"real_list2":real_list2,"pred_list2":pred_list2,"real_list3":real_list3,"pred_list3":pred_list3,"real_list4":real_list4,"pred_list4":pred_list4,"real_list5":real_list5,"pred_list5":pred_list5,"real_list6":real_list6,"pred_list6":pred_list6,"real_list7":real_list7,"pred_list7":pred_list7})
def process_data():
    # cols = ['difficulty', 'gaslimit', 'gasused', 'gaspricel1','confirmtime','timespan1']
    # cols = ['difficulty', 'gaslimit', 'rate', 'gaspricel1', 'confirmtime']
    # data_df = pd.read_csv("./tran_data.csv")
    # # cols=['difficulty','gaspricel1','gaspricel2']
    # X=data_df[cols]
    # y=data_df['gasprice']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # arr_xtrain=np.array(X_train)
    # arr_xtest=np.array(X_test)
    # arr_ytrain=np.array(y_train)
    # arr_ytest=np.array(y_test)
    # # print(arr_xtrain)
    # return arr_xtrain,arr_xtest,arr_ytrain,arr_ytest
    dataset = pd.read_csv('./data.csv', header=0, index_col=0)
    values = dataset.values
    values = values.astype('float32')
    scaledData = scaler.fit_transform(values)
    X = scaledData[:, :-1]
    y = scaledData[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
def process_test(test):
    min = [[8.000000e-01,2.650680e+15, 2.100000e+04, 7.785000e+01,  1.000000e+00, 1.000000e+00,645, 3.0]]
    max = [[ 8.000000e+01,2.797250e+15, 8.000000e+06, 1.533000e+02, 1.305820e+05, 2.240000e+02, 47177, 3.31]]
    max = np.array(max)
    min = np.array(min)
    test = np.array(test)
    test = (test - min) / (max - min)
    # test = np.array(test)
    return test
def predict_time(test):
    dataset = pd.read_csv('./transactioninfo3.csv', header=0, index_col=0)
    values = dataset.values
    values = values.astype('float32')
    scaledData = scaler2.fit_transform(values)
    min = [[2.65068, 21000, 76.43, 1.0]]
    max = [[2.8008, 7607448, 155.65, 200.0]]
    max = np.array(max)
    min = np.array(min)
    test = np.array(test)
    test = (test - min) / (max - min)
    keras.backend.clear_session()
    model_lstm = load_model('./lstm_model_time.h5')
    test = [test]
    test = np.array(test)
    # print(test)
    pre = model_lstm.predict(test)
    x_test = test.reshape((test.shape[0], test.shape[2]))
    y_predict_test = pre.reshape(-1, 1)
    inv_yhat = np.concatenate((x_test, y_predict_test), axis=1)
    inv_yhat = scaler2.inverse_transform(inv_yhat)
    y_pred = inv_yhat[:, -1]
    return (y_pred[0])

def predict_data(test,type):
    X_train, X_test, y_train, y_test=process_data()
    if type=="1":
        # model_xgb = xgb.XGBRegressor(learning_rate=0.01, n_estimators=550, max_depth=7,min_child_weight=1)
        # model_xgb.fit(X_train, y_train)
        # y_pred = model_xgb.predict(test)
        model_xgb= xgb.Booster(model_file='./xgb.model')
        test=process_test(test)
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
        with open("./model_rf.pkl", "rb") as f:
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
        with open("./model_etr.pkl", "rb") as f:
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
        with open("./model_gbr.pkl", "rb") as f:
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
        with open("./model_linear.pkl", "rb") as f:
            model_liner = pickle.load(f)
        # model_liner = LinearRegression()
        # model_liner.fit(X_train,y_train)
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
        with open("./model_bag.pkl", "rb") as f:
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
        model_lstm = load_model('./lstm_model_price.h5')
        test = process_test(test)
        test = [test]
        test = np.array(test)
        # print(test)
        pre = model_lstm.predict(test)
        x_test = test.reshape((test.shape[0], test.shape[2]))
        y_predict_test = pre.reshape(-1, 1)
        inv_yhat = np.concatenate((x_test, y_predict_test), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        y_pred = inv_yhat[:, -1]
    return (y_pred[0])

def pre(request):
    if request.session.get('is_login') == '1':
        uname=request.session['username']
    else:
        uname=''
    type=''
    contime=20
    # price=20
    blockheight = get_recent_block()
    block_list = creat_block_list(blockheight, 2)
    lowprice_list, aveprice_list, maxprice_list,timestamp_list = get_pre_Data(block_list)
    gaspricel1=lowprice_list[0]
    # gaspricel2=lowprice_list[1]
    std=aveprice_list[0]
    maxprice=maxprice_list[0]
    timespan1=timestamp_list[0]-timestamp_list[1]
    block = api.get_block_by_number(blockheight)
    difficulty=int(block['difficulty'],16)
    gaslimit=int(block['gasLimit'],16)
    gasused=int(block['gasUsed'],16)
    rate=round((gasused/gaslimit)*100,2)
    transaction_num=len(block['transactions'])
    size = int(block['size'], 16)
    url = 'https://api.etherscan.io/api?module=block&action=getblockreward&blockno=' + str(blockheight)+ '&apikey=YourApiKeyToken'
    r = requests.get(url)
    response_dict = r.json()
    if not response_dict['result']['blockReward'] :
        reward=0
    else:
        reward = int(response_dict['result']['blockReward']) / 10 ** 18
        reward = round(reward, 2)
    price=gaspricel1
    if request.method == "POST":
        flag=1
        time=int(request.POST.get("confirmtime",None))
        gaslimit=int(request.POST.get("gaslimit",None))
        type=request.POST.get("type",None)
        gasused=rate*gaslimit/100
        ethusd=getrate()
        ethusd=float(ethusd)
        test = np.array([[gaspricel1,difficulty, gaslimit,ethusd, time,transaction_num,size,reward]])
        price=predict_data(test,type)
        if type!="8":
            price=abs(round(price,2))
            if price> std:
                std=price
            if price>maxprice:
                maxprice=price
            contime=time
        else:
            std=0
            maxprice=0
            contime=0
    else:
        flag=0
        gaslimit=21000
        type='1'
    return render(request,'pre.html',{"flag":flag,"gaslimit":gaslimit,"height":blockheight,"rate":rate,"confirm":contime,"price":price,"gaspricel1":gaspricel1,"std":std,"maxprice":maxprice,"uname":uname,"timespan":timespan1,"type":type,"transactionnum":transaction_num})
def home(request):
    num2 = 5
    if request.method == "POST":
        num2 = int(request.POST.get("num2", None))
    blockheight = get_recent_block()
    block_list = creat_block_list(blockheight, num2)
    id_list, num_list, lowprice_list, aveprice_list,maxprice_list = get_tran_Data(block_list)
    return render(request, 'clara/index.html',
                 {"label": id_list, "height": block_list, "num": num_list, "lowprice": lowprice_list,
                   "aveprice": aveprice_list,"maxprice":maxprice_list})
    #return render(request,"clara/index.html")

def homedata(request):
    num2 = 5
    if request.method == "POST":
        num2 = int(request.POST.get("num2", None))
    blockheight = get_recent_block()
    block_list = creat_block_list(blockheight, num2)
    id_list, num_list, lowprice_list, aveprice_list,maxprice_list = get_tran_Data(block_list)
    return render(request, 'provence/index.html',
                  {"label": id_list, "height": block_list, "num": num_list, "lowprice": lowprice_list,"aveprice": aveprice_list,"maxprice":maxprice_list})
def about(request):
    return render(request,"clara/about.html")
def forecast(request):
    return render(request,"clara/forecast.html")
def backtest(request):
    return render(request,"clara/backtest.html")
def contact(request):
    return render(request,"clara/contact.html")
def dataPage(request):
    date1, time1, value1 = fetch_info(50, "https://etherscan.io/chart/gaslimit?output=csv", "./gaslimit.csv")
    date2, time2, value2 = fetch_info(50, "https://etherscan.io/chart/gasused?output=csv", "./gasused.csv")
    date3, time3, value3 = fetch_info(50, "https://etherscan.io/chart/gasprice?output=csv", "./gasprice.csv")
    date4, time4, value4 = fetch_info(50, "https://etherscan.io/chart/blocktime?output=csv", "./blocktime.csv")
    return render(request, 'clara/data.html',
                  {"date1": date1, "time1": time1, "value1": value1, "date2": date2, "time2": time2, "value2": value2,
                   "date3": date3, "time3": time3, "value3": value3, "date4": date4, "time4": time4, "value4": value4})
    #return render(request,"clara/data.html")
def blockexplorer(request):
    num2 = 20
    infotype = request.GET.get("type", None)
    info = request.GET.get("info", None)
    if infotype == None:
        blockheight = get_recent_block()
        block_list = creat_block_list(blockheight, num2)
        id_list, num_list, lowprice_list, aveprice_list ,maxprice_list= get_tran_Data(block_list)
        print("ok")
        return render(request, 'block-explorer.html', {"height": block_list, "num": num_list, "lowprice": lowprice_list,"aveprice": aveprice_list})
    else:
        if infotype=='block':
            list = get_block_info(int(info))
            return render(request, 'block-explorer.html', {"type": infotype, "info": list})
        else:
            list = get_tran_info(info)
            path = os.path.abspath('.')
            return render(request, 'block-explorer.html', {"type": infotype, "info": list, "path": path})

def blockexplorerdata(request):
    num2 = 20
    infotype = request.GET.get("type", None)
    info = request.GET.get("info", None)
    if infotype == None:
        blockheight = get_recent_block()
        block_list = creat_block_list(blockheight, num2)
        id_list, num_list, lowprice_list, aveprice_list, maxprice_list = get_tran_Data(block_list)
        print("ok")
        return render(request, 'blockexplorerdata.html',{"height": block_list, "num": num_list, "lowprice": lowprice_list, "aveprice": aveprice_list})
    else:
        if infotype == 'block':
            list = get_block_info(int(info))
            return render(request, 'blockexplorerdata.html', {"type": infotype, "info": list})
        else:
            list = get_tran_info(info)
            path = os.path.abspath('.')
            return render(request, 'blockexplorerdata.html', {"type": infotype, "info": list, "path": path})

def data(request):
    infotype = request.GET.get("type", None)
    info = request.GET.get("info", None)
    if infotype == 'block':
        print(info)
        list = get_block_info(int(info))
        return render(request, 'data.html', {"type": infotype, "info": list})
    else:
        list = get_tran_info(info)
        path = os.path.abspath('.')
        return render(request, 'data.html', {"type": infotype, "info": list, "path": path})
def datal(request):
    infotype = request.GET.get("type", None)
    info = request.GET.get("info", None)
    if infotype == 'block':
        print(info)
        list = get_block_info(int(info))
        return render(request, 'datal.html', {"type": infotype, "info": list})
    else:
        list = get_tran_info(info)
        path = os.path.abspath('.')
        return render(request, 'datal.html', {"type": infotype, "info": list, "path": path})
def gasapi(request):
    time = request.GET.get("confirmtime", None)
    gaslimit = request.GET.get("gaslimit", None)
    price = request.GET.get("gasprice", None)
    blockheight = get_recent_block()
    block_list = creat_block_list(blockheight, 2)
    lowprice_list, aveprice_list, maxprice_list, timestamp_list = get_pre_Data(block_list)
    gaspricel1 = lowprice_list[0]
    timespan1 = timestamp_list[0] - timestamp_list[1]
    block = api.get_block_by_number(blockheight)
    difficulty = int(block['difficulty'], 16)
    # gaslimitl1 = int(block['gasLimit'], 16)
    # gasusedl1 = int(block['gasUsed'], 16)
    transaction_num = len(block['transactions'])
    # rate = round((gasusedl1 / gaslimitl1) * 100, 2)
    ethusd = getrate()
    ethusd = float(ethusd)
    size = int(block['size'], 16)
    url = 'https://api.etherscan.io/api?module=block&action=getblockreward&blockno=' + str(blockheight) + '&apikey=YourApiKeyToken'
    r = requests.get(url)
    response_dict = r.json()
    reward = int(response_dict['result']['blockReward']) / 10 ** 18
    reward = round(reward, 2)
    # gasused = rate * gaslimit / 100
    if time != None and gaslimit != None:
        time = int(time)
        gaslimit = int(gaslimit)
        test = np.array([[gaspricel1,difficulty, gaslimit, ethusd, time,transaction_num,size,reward]])
        gasprice = predict_data(test,'1')
        gasprice = round(gasprice, 2)
        return JsonResponse({'gasprice': gasprice, 'confirmtime': time, 'gaslimit': gaslimit, 'message': "predict gasprice",'model':'xgboost'})
    elif price != None and gaslimit != None:
        price=int(price)
        gaslimit = int(gaslimit)
        test = np.array([[difficulty, gaslimit, ethusd, price]])
        confirm = predict_time(test)
        # confirm = str(round(confirm, 2))
        confirm = round(confirm, 2)
        # confirm = price + int(gaslimit)
        return JsonResponse({'confimetime': confirm, 'gasprice': price, 'gaslimit': gaslimit, 'message': "predict confirmtime","model":"LSTM"})
    else:
        return JsonResponse({"status": "0", "message": "NOTOK", "result": "Invalid API URL endpoint, use api.gaspricing.io"})
#Newly added information about chart data
def fetch_info(num,urls,outputfile):
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0;WOW64;rv:64.0) Gecko/20100101 Firefox/64.0'}
    req = urllib.request.Request(url=urls, headers=headers)
    data=urllib.request.urlopen(req).read()
    with open(outputfile, "wb") as code:
        code.write(data)
    df_tran = pd.read_csv(outputfile)
    num=-num
    df_new = df_tran.iloc[num:, :]
    date = df_new["Date(UTC)"].tolist()
    timestamps = df_new['UnixTimeStamp'].tolist()
    value = df_new['Value'].tolist()
    return  date,timestamps,value
def chart(request):
    date1, time1, value1 = fetch_info(100,"https://etherscan.io/chart/gaslimit?output=csv", "./gaslimit.csv")
    date2, time2, value2 = fetch_info(100,"https://etherscan.io/chart/gasused?output=csv", "./gasused.csv")
    date3, time3, value3 = fetch_info(100,"https://etherscan.io/chart/gasprice?output=csv", "./gasprice.csv")
    date4, time4, value4 = fetch_info(100,"https://etherscan.io/chart/blocktime?output=csv", "./blocktime.csv")
    return render(request, 'chart.html', {"date1": date1, "time1": time1, "value1": value1,"date2":date2,"time2":time2,"value2":value2,"date3":date3,"time3":time3,"value3":value3,"date4":date4,"time4":time4,"value4":value4})
def homedata(request):
    num2 = 5
    if request.method == "POST":
        num2 = int(request.POST.get("num2", None))
    blockheight = get_recent_block()
    block_list = creat_block_list(blockheight, num2)
    id_list, num_list, lowprice_list, aveprice_list,maxprice_list = get_tran_Data(block_list)
    return render(request, 'provence/index.html',{"label": id_list, "height": block_list, "num": num_list, "lowprice": lowprice_list,"aveprice": aveprice_list,"maxprice":maxprice_list})
def phome(request):
    return render(request,"provence/index.html")
def pabout(request):
    return render(request,"provence/about.html")
def pforecast(request):
    return render(request,"provence/forecast.html")
def pbacktest(request):
    return render(request,"provence/backtest.html")
def pcontact(request):
    return render(request,"provence/contact.html")
def pdataPage(request):
    date1, time1, value1 = fetch_info(50, "https://etherscan.io/chart/gaslimit?output=csv", "./gaslimit.csv")
    date2, time2, value2 = fetch_info(50, "https://etherscan.io/chart/gasused?output=csv", "./gasused.csv")
    date3, time3, value3 = fetch_info(50, "https://etherscan.io/chart/gasprice?output=csv", "./gasprice.csv")
    date4, time4, value4 = fetch_info(50, "https://etherscan.io/chart/blocktime?output=csv", "./blocktime.csv")
    return render(request, 'provence/data.html',{"date1": date1, "time1": time1, "value1": value1, "date2": date2, "time2": time2, "value2": value2,
                   "date3": date3, "time3": time3, "value3": value3, "date4": date4, "time4": time4, "value4": value4})
