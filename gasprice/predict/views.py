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
def get_price_Data(block_list):
    lowprice_list=[] #low price
    for blockheight in block_list:
        block = api.get_block_by_number(blockheight)
        if len(block['transactions'])==0:
            lowprice_list.append(0)
        else:
            gasprice=[]
            for row in block['transactions']:
                gasprice.append(round(int(row['gasPrice'],16)/1000000000,2))
            lowprice_list.append(min(gasprice))
    return lowprice_list
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
    count=test_set.shape[0]
    # train set
    cols = ['height', 'date', 'gaspricel1','difficulty', 'gaslimit', 'rate',  'confirmtime', 'transaction_count','size', 'reward']
    X = dataset[cols]
    y = dataset['gasprice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    n_train_size = X_train.shape[0]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train = y_train.reshape(-1, 1)
    train_set = np.concatenate((X_train, y_train), axis=1)
    test_set = np.array(test_set)
    all = np.concatenate((train_set, test_set), axis=0)
    values = all[:, 2:]
    scaledData = scaler1.fit_transform(values)
    train = scaledData[:n_train_size, :]
    test = scaledData[n_train_size:, :]
    # split into input and outputs
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]
    return  X_train, X_test, y_train, y_test,count
def eval(request):
    if request.method == "POST":
        start=int(request.POST.get("time",None))
        type=request.POST.get("type",None)
        stop=start+20000
        flag=1
    else:
        type="1"
        start=26100000
        stop=26120000
        flag=0
    X_train, X_test, y_train, y_test,count1 = process_eval(start,stop)
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
        elif type == "2":
            with open("./model_rf.pkl", "rb") as f:
                model = pickle.load(f)
            y_predict_test = model.predict(X_test)
        elif type == "3":
            with open("./model_etr.pkl", "rb") as f:
                model = pickle.load(f)
            y_predict_test = model.predict(X_test)
        elif type == "4":
            with open("./model_gbr.pkl", "rb") as f:
                model= pickle.load(f)
            y_predict_test = model.predict(X_test)
        elif type == "5":
            with open("./model_linear.pkl", "rb") as f:
                model= pickle.load(f)
            y_predict_test = model.predict(X_test)
        elif type == "6":
            with open("./model_bag.pkl", "rb") as f:
                model= pickle.load(f)
            y_predict_test = model.predict(X_test)
        # # 测试集
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
        if count1<30:
            id_list.append(id+1)
        elif count1<100:
            if (id + 1) % 5 == 0:
                id_list.append(id + 1)
            else:
                id_list.append('')
        else:
            if (id + 1) % 10 == 0:
                id_list.append(id + 1)
            else:
                id_list.append('')
    label_list = ['<1', '1<5', '5<10', '10<15', '>15']
    return render(request,'eval.html',{"flag":flag,"type":type,"time":start,"count":count1,"label":json.dumps(id_list),"List1":json.dumps(real_list),"List2":json.dumps(pred_list),"List3":json.dumps(error_list),"label1":label_list,"List4":count_list})
def process_data():
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
    elif type=="3":
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
    elif type=="4":
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
    elif type=="5":
        with open("./model_linear.pkl", "rb") as f:
            model_liner = pickle.load(f)
        test = process_test(test)
        test = np.array(test)
        pre = model_liner.predict(test)
        x_test = test.reshape((test.shape[0], test.shape[1]))
        y_predict_test = pre.reshape(-1, 1)
        inv_yhat = np.concatenate((x_test, y_predict_test), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        y_pred = inv_yhat[:, -1]
    elif type=="6":
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
    elif type=="7":
        keras.backend.clear_session()
        model_lstm = load_model('./lstm_model_price.h5')
        test = process_test(test)
        test = [test]
        test = np.array(test)
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
    blockheight = get_recent_block()
    block_list = creat_block_list(blockheight, 1)
    lowprice_list= get_price_Data(block_list)
    gaspricel1=lowprice_list[0]
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
    # price=gaspricel1
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
            contime=time
    else:
        flag=0
        gaslimit=''
        type=''
        price=''
    return render(request,'pre.html',{"flag":flag,"gaslimit":gaslimit,"height":blockheight,"rate":rate,"confirm":contime,"price":price,"gaspricel1":gaspricel1,"uname":uname,"type":type,"transactionnum":transaction_num})
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
    block_list = creat_block_list(blockheight, 1)
    lowprice_list= get_price_Data(block_list)
    gaspricel1 = lowprice_list[0]
    block = api.get_block_by_number(blockheight)
    difficulty = int(block['difficulty'], 16)
    transaction_num = len(block['transactions'])
    ethusd = getrate()
    ethusd = float(ethusd)
    size = int(block['size'], 16)
    url = 'https://api.etherscan.io/api?module=block&action=getblockreward&blockno=' + str(blockheight) + '&apikey=YourApiKeyToken'
    r = requests.get(url)
    response_dict = r.json()
    reward = int(response_dict['result']['blockReward']) / 10 ** 18
    reward = round(reward, 2)
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
        confirm = round(confirm, 2)
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
