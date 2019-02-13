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
import  urllib
from urllib.request import urlretrieve
from keras.models import load_model
# 全局变量
with open('./api_key.json', mode='r') as key_file:
    key = json.loads(key_file.read())['key']
api = Proxies(api_key=key)
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
def eval(request):
    if request.session.get('is_login') == '1':
        uname = request.session['username']
    else:
        uname = ''
    if request.method == "POST":
        count1=int(request.POST.get("count",None))
        data1 = Predictres.objects.filter(id__lte=count1, id__gte=1)
    else:
        count1=20
        data1 = Predictres.objects.filter(id__lte=20, id__gte=1)
    real_list = []
    pred_list = []
    error_list = []
    id_list=[]
    cursor = connection.cursor()
    cursor.execute("select * from predictres where id>0 limit %s" % (count1))
    # 使用一个变量来接收查询到的数据，
    # fetchall（）返回查询到的所有数据
    results = cursor.fetchall()
    for row in results:
        id = row[0]
        real = row[1]
        pred=row[2]
        error=row[3]
        id_list.append(id)
        real_list.append(real)
        pred_list.append(pred)
        error_list.append(error)
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=1 group by i_p;")
    results1 = cursor.fetchall()
    label_list = ['<1', '1<5', '5<10', '10<15', '>15']
    count_list=[]
    for row in results1:
        trancount = row[1]
        count_list.append(trancount)
    return render(request,'eval.html',{"data": data1,"label":json.dumps(id_list),"List1":json.dumps(real_list),"List2":json.dumps(pred_list),"List3":json.dumps(error_list),"uname":uname,"label1":label_list,"List4":count_list})
def modeleval(request):
    cursor = connection.cursor()
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=1 group by i_p;")
    results1 = cursor.fetchall()
    label_list = ['<1', '1<5', '5<10', '10<15', '>15']
    count_list=[]
    for row in results1:
        trancount = row[1]
        count_list.append(trancount)
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=2 group by i_p;")
    results2 = cursor.fetchall()
    count_list2 = []
    for row in results2:
        trancount = row[1]
        count_list2.append(trancount)
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=3 group by i_p;")
    results3 = cursor.fetchall()
    count_list3 = []
    for row in results3:
        trancount = row[1]
        count_list3.append(trancount)
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=4 group by i_p;")
    results4 = cursor.fetchall()
    count_list4 = []
    for row in results4:
        trancount = row[1]
        count_list4.append(trancount)
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=5 group by i_p;")
    results5 = cursor.fetchall()
    count_list5 = []
    for row in results5:
        trancount = row[1]
        count_list5.append(trancount)
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=6 group by i_p;")
    results6 = cursor.fetchall()
    count_list6 = []
    for row in results6:
        trancount = row[1]
        count_list6.append(trancount)
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres where uid=7 group by i_p;")
    results7 = cursor.fetchall()
    count_list7 = []
    for row in results7:
        trancount = row[1]
        count_list7.append(trancount)
    return render(request,'modeleval.html',{"label1":label_list,"List1":count_list,"List2":count_list2,"List3":count_list3,"List4":count_list4,"List5":count_list5,"List6":count_list6,"List7":count_list7})
def process_data():
    # cols = ['difficulty', 'gaslimit', 'gasused', 'gaspricel1','confirmtime','timespan1']
    cols = ['difficulty', 'gaslimit', 'rate', 'gaspricel1', 'confirmtime']
    data_df = pd.read_csv("./tran_data.csv")
    # cols=['difficulty','gaspricel1','gaspricel2']
    X=data_df[cols]
    y=data_df['gasprice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    arr_xtrain=np.array(X_train)
    arr_xtest=np.array(X_test)
    arr_ytrain=np.array(y_train)
    arr_ytest=np.array(y_test)
    # print(arr_xtrain)
    return arr_xtrain,arr_xtest,arr_ytrain,arr_ytest




def predict_data(test,type):
    X_train, X_test, y_train, y_test=process_data()
    if type=="1":
        model_xgb = xgb.XGBRegressor(learning_rate=0.01, n_estimators=550, max_depth=7,min_child_weight=1)
        model_xgb.fit(X_train, y_train)
        y_pred = model_xgb.predict(test)
    elif type=="2":
        model_rfr = RandomForestRegressor(n_estimators=550)
        model_rfr.fit(X_train,y_train)
        y_pred=model_rfr.predict(test)
    elif type=="3":
        model_etr = ExtraTreesRegressor(n_estimators=100)
        model_etr.fit(X_train,y_train)
        y_pred=model_etr.predict(test)
    elif type=="4":
        model_gbr=GradientBoostingRegressor(n_estimators=550)
        model_gbr.fit(X_train,y_train)
        y_pred=model_gbr.predict(test)
    elif type=="5":
        model_liner = LinearRegression()
        model_liner.fit(X_train,y_train)
        y_pred=model_liner.predict(test)
    elif type=="6":
        model_nn = load_model('./my_model.h5')
        mean=[2.72852106e+15 ,3.88085648e+05 ,1.14774505e+02 ,3.51060848e+00,9.88294294e+02]
        std=[3.54962054e+13, 9.77509992e+05, 1.05278795e+01 ,5.24377910e+00,5.47060186e+03]
        mean=np.array(mean)
        std=np.array(std)
        test-=mean
        test/=std
        y_pred=model_nn.predict(test)
        y_pred=y_pred.ravel()
    elif type=="7":
        model_bag=BaggingRegressor(tree.DecisionTreeRegressor(), n_estimators=100, max_samples=0.3)
        model_bag.fit(X_train,y_train)
        y_pred=model_bag.predict(test)
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
    price=gaspricel1
    if request.method == "POST":
        time=int(request.POST.get("confirmtime",None))
        gaslimit=int(request.POST.get("gaslimit",None))
        type=request.POST.get("type",None)
        gasused=rate*gaslimit/100
        ethusd=getrate()
        ethusd=float(ethusd)
        test = np.array([[difficulty, gaslimit,ethusd, gaspricel1,time]])
        price=predict_data(test,type)
        price=round(price,2)
        if price> std:
            std=price
        if price>maxprice:
            maxprice=price
        contime=time
    return render(request,'pre.html',{"height":blockheight,"rate":rate,"confirm":contime,"price":price,"gaspricel1":gaspricel1,"std":std,"maxprice":maxprice,"uname":uname,"timespan":timespan1,"type":type})
def home(request):
    num2 = 5
    if request.method == "POST":
        num2 = int(request.POST.get("num2", None))
    blockheight = get_recent_block()
    block_list = creat_block_list(blockheight, num2)
    id_list, num_list, lowprice_list, aveprice_list,maxprice_list = get_tran_Data(block_list)
    return render(request, 'clara/index.html',
                 {"label": id_list, "height": block_list, "num": num_list, "lowprice": lowprice_list,
                   "aveprice": aveprice_list})
    #return render(request,"clara/index.html")

def homedata(request):
    num2 = 5
    if request.method == "POST":
        num2 = int(request.POST.get("num2", None))
    blockheight = get_recent_block()
    block_list = creat_block_list(blockheight, num2)
    id_list, num_list, lowprice_list, aveprice_list,maxprice_list = get_tran_Data(block_list)
    return render(request, 'provence/index.html',
                  {"label": id_list, "height": block_list, "num": num_list, "lowprice": lowprice_list,
                   "aveprice": aveprice_list})
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
    time = request.GET.get("time", None)
    gaslimit = request.GET.get("gaslimit", None)
    price = request.GET.get("price", None)
    if time != None and gaslimit != None:
        time = int(time)
        gaslimit = int(gaslimit)
        blockheight = get_recent_block()
        block_list = creat_block_list(blockheight, 2)
        lowprice_list, aveprice_list, maxprice_list, timestamp_list = get_pre_Data(block_list)
        gaspricel1 = lowprice_list[0]
        timespan1 = timestamp_list[0] - timestamp_list[1]
        block = api.get_block_by_number(blockheight)
        difficulty = int(block['difficulty'], 16)
        gaslimitl1 = int(block['gasLimit'], 16)
        gasusedl1 = int(block['gasUsed'], 16)
        rate = round((gasusedl1 / gaslimitl1) * 100, 2)
        ethusd=getrate()
        ethusd=float(ethusd)
        gasused = rate * gaslimit / 100
        test = np.array([[difficulty, gaslimit, ethusd, gaspricel1, time]])
        gasprice = predict_data(test,'1')
        gasprice = str(round(gasprice, 2))
        return JsonResponse({'gasprice': gasprice, 'confirmtime': time, 'gaslimit': gaslimit, 'message': "predict gasprice",'model':'xgboost'})
    elif price != None and gaslimit != None:
        confirm = int(price) + int(gaslimit)
        return JsonResponse({'confimetime': confirm, 'gasprice': price, 'gaslimit': gaslimit, 'message': "predict confirmtime"})
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
    return render(request, 'provence/index.html',
                  {"label": id_list, "height": block_list, "num": num_list, "lowprice": lowprice_list,
                   "aveprice": aveprice_list})
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
    return render(request, 'provence/data.html',
                  {"date1": date1, "time1": time1, "value1": value1, "date2": date2, "time2": time2, "value2": value2,
                   "date3": date3, "time3": time3, "value3": value3, "date4": date4, "time4": time4, "value4": value4})
