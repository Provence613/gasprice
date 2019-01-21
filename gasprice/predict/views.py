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
import xgboost as xgb
import pandas as pd
import numpy as np
import os
from etherscan.proxies import Proxies
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
            aveprice_list.append(0)
        else:
            gasprice=[]
            for row in block['transactions']:
                gasprice.append(round(int(row['gasPrice'],16)/1000000000,2))
            # print("The min gasprice is {} in block {}".format(min(gasprice),blockheight))
            lowprice_list.append(min(gasprice))
            aveprice_list.append(round(np.mean(gasprice),2))
        i=i+1
        # height_list.append(blockheight)
        # print(int(block['transactions'][1]['gasPrice'],16))
    return id_list,num_list,lowprice_list,aveprice_list
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
def detailtx(request):
    tx_hash=request.GET.get("hash",None)
    list=get_tran_info(tx_hash)
    return render(request, 'detailtx.html',{"tran":list})
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
def detailblock(request):
    blocknumber=int(request.GET.get("num",None))
    list=get_block_info(blocknumber)
    return render(request, 'detailblock.html',{"block":list})
def data(request):
    num2=50
    if request.method == "POST":
        num2 = int(request.POST.get("num2",None))
    blockheight=get_recent_block()
    block_list=creat_block_list(blockheight,num2)
    id_list,num_list, lowprice_list, aveprice_list=get_tran_Data(block_list)
    return render(request, 'data.html', {"label":id_list,"height":block_list,"num":num_list,"lowprice":lowprice_list,"aveprice":aveprice_list})
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
def show(request):
    if request.session.get('is_login') == '1':
        uname = request.session['username']
    else:
        uname = ''
    tran_list = []
    id_list = []
    price_list=[]
    count_list=[]
    time_list=[]
    count1_list=[]
    cursor = connection.cursor()
    num1=20
    cursor.execute("select INTERVAL(gasprice,1,5,15,30) as i_p,COUNT(height) from transinfo group by i_p")
    # cursor.execute("SELECT gasprice,COUNT('height') from transinfo GROUP BY gasprice limit %s" % (num1))
    # 使用一个变量来接收查询到的数据，
    # fetchall（）返回查询到的所有数据
    results = cursor.fetchall()
    price_list=['<1','1<5','5<15','15<30','>30']
    for row in results:
        # gasprice = row[0]
        trancount = row[1]
        # price_list.append(gasprice)
        count_list.append(trancount)
    cursor.execute("select count(*),timestamp from transinfo GROUP BY timestamp")
    results1 = cursor.fetchall()
    for row in results1:
        timestamp = row[1]
        trancount = row[0]
        time_list.append(timestamp)
        count1_list.append(trancount)
    if request.method == "POST":
        num=int(request.POST.get("num",None))
        # for i in Transinfo.objects.filter(id__lt=num):
        #     id_list.append(i.id)
        #     tran_list.append(i.gasprice)
    else:
        num=15
    cursor.execute("select gasprice from transinfo order by height DESC LIMIT  %s" % (num))
    results2 = cursor.fetchall()
    j=1
    for row in results2:
        price= row[0]
        tran_list.append(price)
        id_list.append(j)
        j=j+1
        # for i in Transinfo.objects.filter(id__lt=7):
        #     id_list.append(i.id)
        #     tran_list.append(i.gasprice)
    return render(request, 'show.html',{"List":json.dumps(tran_list),"label":json.dumps(id_list),"List1":json.dumps(count_list),"label1":json.dumps(price_list),"List2":json.dumps(count1_list),"label2":json.dumps(time_list),'uname':uname})
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
    cursor.execute("select INTERVAL(error,1,5,10,15) as i_p,COUNT(id) from predictres group by i_p;")
    results1 = cursor.fetchall()
    label_list = ['<1', '1<5', '5<10', '10<15', '>15']
    count_list=[]
    for row in results1:
        trancount = row[1]
        count_list.append(trancount)
    return render(request,'eval.html',{"data": data1,"label":json.dumps(id_list),"List1":json.dumps(real_list),"List2":json.dumps(pred_list),"List3":json.dumps(error_list),"uname":uname,"label1":label_list,"List4":count_list})
def process_data():
    cols = ['difficulty', 'gaslimit', 'gasused', 'gaspricel1','confirmtime','timespan1']
    data_df = pd.read_csv("./transaction2.csv")
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




def predict_data(test):
    X_train, X_test, y_train, y_test=process_data()
    model = xgb.XGBRegressor(learning_rate=0.2, n_estimators=550, max_depth=3,min_child_weight=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(test)
    return (y_pred[0])

def pre(request):
    if request.session.get('is_login') == '1':
        uname=request.session['username']
    else:
        uname=''
    contime=20
    # price=20
    blockheight = get_recent_block()
    block_list = creat_block_list(blockheight, 2)
    lowprice_list, aveprice_list, maxprice_list,timestamp_list = get_pre_Data(block_list)
    gaspricel1=lowprice_list[0]
    gaspricel2=lowprice_list[1]
    std=aveprice_list[0]
    maxprice=maxprice_list[0]
    timespan1=timestamp_list[0]-timestamp_list[1]
    block = api.get_block_by_number(blockheight)
    difficulty=int(block['difficulty'],16)
    gaslimit=int(block['gasLimit'],16)
    gasused=int(block['gasUsed'],16)
    # cursor = connection.cursor()
    # cursor.execute("SELECT * FROM transinfo ORDER BY height DESC LIMIT 1")
    # # 使用一个变量来接收查询到的数据，
    # # fetchall（）返回查询到的所有数据
    # results = cursor.fetchall()
    # for row in results:
    #     gasprice=row[1]
    #     height= row[2]
    #     gaslimit = row[4]
    #     gasused = row[5]
    #     std=row[7]
    #     difficulty=row[3]
    #     gaspricel1=row[8]
    #     gaspricel2=row[9]
    #     maxprice=row[10]
    rate=round((gasused/gaslimit)*100,2)
    price=gaspricel1
    if request.method == "POST":
        time=int(request.POST.get("confirmtime",None))
        gaslimit=int(request.POST.get("gaslimit",None))
        gasused=rate*gaslimit/100
        test = np.array([[difficulty, gaslimit, gasused, gaspricel1,time,timespan1]])
        price=predict_data(test)
        price=round(price,2)
        if price> std:
            std=price
        if price>maxprice:
            maxprice=price
        contime=time
    return render(request,'pre.html',{"height":blockheight,"rate":rate,"confirm":contime,"price":price,"std":std,"maxprice":maxprice,"uname":uname,"timespan":timespan1})
def login(request):
    errors = []
    account = None
    password = None
    if request.method == "POST":
        if not request.POST.get('username'):
            errors.append('username is not null')
        else:
            account = request.POST.get('username')

        if not request.POST.get('pass'):
            errors = request.POST.get('password is not null')
        else:
            password = request.POST.get('pass')

        if account is not None and password is not None:
            user = User.objects.filter(username=account, password=password)
            if user:
                request.session['is_login'] = '1'  # 这个session是用于后面访问每个页面（即调用每个视图函数时要用到，即判断是否已经登录，用此判断）
                request.session['user_id'] = user[0].id
                request.session['username']=user[0].username
                return HttpResponseRedirect('/pre')
            else:
                errors.append('username or password is not exist')
    return  render(request,"login.html",{"errors":errors})
def register(request):
    errors = []
    account = None
    password = None
    password2 = None
    email = None
    company=None
    CompareFlag = False

    if request.method == 'POST':
        if not request.POST.get('username'):
            errors.append('username is not null')
        else:
            account = request.POST.get('username')

        if not request.POST.get('pass'):
            errors.append('password is not null')
        else:
            password = request.POST.get('pass')
        if not request.POST.get('repass'):
            errors.append('confirm password is noy null')
        else:
            password2 = request.POST.get('repass')
        if not request.POST.get('email'):
            errors.append('email is not null')
        else:
            email = request.POST.get('email')
        if not request.POST.get('company'):
            errors.append('company is not null')
        else:
            company = request.POST.get('company')
        if User.objects.filter(username=account).exists():
            errors.append('user has already existed')
        else:
            if password is not None:
                 if password == password2:
                      CompareFlag = True
                 else:
                    errors.append('The password is not same')
            if account is not None and password is not None and password2 is not None and email is not None and CompareFlag:
                user = User(username=account,password=password,email=email,company=company)
                user.save()
            # userlogin = auth.authenticate(username=account, password=password)
            # auth.login(request, userlogin)
            return HttpResponseRedirect('/login')

    return  render(request,"register.html",{"errors":errors})
def logout(request):
    request.session.flush()
    return HttpResponseRedirect('/pre')
def home(request):
    num2 = 10
    if request.method == "POST":
        num2 = int(request.POST.get("num2", None))
    blockheight = get_recent_block()
    block_list = creat_block_list(blockheight, num2)
    id_list, num_list, lowprice_list, aveprice_list = get_tran_Data(block_list)
    return render(request, 'home.html',
                  {"label": id_list, "height": block_list, "num": num_list, "lowprice": lowprice_list,
                   "aveprice": aveprice_list})
    #return render(request,"home.html")
<<<<<<< HEAD
def homedata(request):
    num2 = 10
    if request.method == "POST":
        num2 = int(request.POST.get("num2", None))
    blockheight = get_recent_block()
    block_list = creat_block_list(blockheight, num2)
    id_list, num_list, lowprice_list, aveprice_list = get_tran_Data(block_list)
    return render(request, 'homedata.html',
                  {"label": id_list, "height": block_list, "num": num_list, "lowprice": lowprice_list,
                   "aveprice": aveprice_list})
=======
>>>>>>> a3232fc00c41f5dc1d05aadb423b89addfaf951e
def about(request):
    return render(request,"about.html")
def forecast(request):
    return render(request,"forecast.html")
def backtest(request):
    return render(request,"backtest.html")
def contact(request):
    return render(request,"contact.html")
def blockexplorer(request):
    num2 = 20
    infotype = request.GET.get("type", None)
    info = request.GET.get("info", None)
    if infotype == None:
        blockheight = get_recent_block()
        block_list = creat_block_list(blockheight, num2)
        id_list, num_list, lowprice_list, aveprice_list = get_tran_Data(block_list)
        print("ok")
        return render(request, 'block-explorer.html', {"height": block_list, "num": num_list, "lowprice": lowprice_list,"aveprice": aveprice_list})
    else:
        if infotype=='block':
            list = get_block_info(info)
            return render(request, 'block-explorer.html', {"type": infotype, "info": list})
        else:
            list = get_tran_info(info)
            path = os.path.abspath('.')
            return render(request, 'block-explorer.html', {"type": infotype, "info": list, "path": path})

<<<<<<< HEAD
def blockexplorerdata(request):
    num2 = 20
    infotype = request.GET.get("type", None)
    info = request.GET.get("info", None)
    if infotype == None:
        blockheight = get_recent_block()
        block_list = creat_block_list(blockheight, num2)
        id_list, num_list, lowprice_list, aveprice_list = get_tran_Data(block_list)
        print("ok")
        return render(request, 'blockexplorerdata.html', {"height": block_list, "num": num_list, "lowprice": lowprice_list,"aveprice": aveprice_list})
    else:
        if infotype=='block':
            list = get_block_info(info)
            return render(request, 'blockexplorerdata.html', {"type": infotype, "info": list})
        else:
            list = get_tran_info(info)
            path = os.path.abspath('.')
            return render(request, 'blockexplorerdata.html', {"type": infotype, "info": list, "path": path})

=======
>>>>>>> a3232fc00c41f5dc1d05aadb423b89addfaf951e
