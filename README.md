# gaspricing
We provide APIs to help users predict gasprice and confirm time.
Using tutorials:
http://127.0.0.1:8000/gasapi/?confirmtime=21&gaslimit=21000   predict gasprice 
result:{"gasprice": 2.41, "confirmtime": 21, "gaslimit": 21000, "message": "predict gasprice", "model": "xgboost"}
http://localhost:8000/gasapi/?price=25&gaslimit=21000  predict confirm time
result:{"confimetime": 10.47, "gasprice": 21, "gaslimit": 21000, "message": "predict confirmtime", "model": "LSTM"}
If the input format is wrong:http://localhost:8000/gasapi/?
result:{"status": "0", "message": "NOTOK", "result": "Invalid API URL endpoint, use api.gaspricing.io"}
