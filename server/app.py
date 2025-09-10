from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pdx    
import numpy as np
app = Flask(__name__)

name = "M9-5-2025-unstable"
model = load_model(f"ML-model/{name}/model.h5")
with open(f"ML-model/{name}/text.txt", "r", encoding="utf-8") as f:
    class_names = f.read().splitlines()

incress = [0,1,2,3,4,5, 10,11,12,13,14,15, 20,21,22,23,24,25, 30,31,32,33,34,35, 40,41,42,43,44,45, 50,51,52,53,54,55]
lc = False
rc = False
ldata = []
rdata = []
forpredict = []
label="Hello"
start = False
status = "Idle"

def predict(data):
    global start, label, forpredict, ldata, rdata, lc, rc, status, incress
    if start:
        pred = model.predict(np.array([data]).reshape(1, -1))
        index = np.argmax(pred)
        label = class_names[index]
        print(label, "<==>")
        forpredict=[]
        status = "Idle"
        start = False
        print(""*3)
        print("Data saved")
        print(""*3)

@app.route('/', methods=['POST', 'GET'])
def mainweb():
    global label, status
    return render_template('index.html', prediction=label, status=status)

@app.route('/btn', methods=['POST'])
def btn():
    global start, status
    status = "Working"
    print("Button pressed")
    if not start:
        start = True
    return "OK"

@app.route('/left', methods=['POST'])
def lget():
    global ldata, lc, rc, start
    # if start:
    lc = True
    data = request.get_json()
    # laccel_x = data["accel_x"]
    # laccel_y = data["accel_y"]
    # laccel_z = data["accel_z"]
    # lgyro_x  = data["gyro_x"]
    # lgyro_y  = data["gyro_y"]
    # lgyro_z  = data["gyro_z"]
    lflex_1 = data["flex_raw_1"]
    lflex_2 = data["flex_raw_2"]*2
    lflex_3 = data["flex_raw_3"]
    lflex_4 = data["flex_raw_4"]
    # lflex_5 = data["flex_raw_5"]
    ldata = [lflex_1, lflex_2, lflex_3, lflex_4]
    print(ldata)
    predict(ldata)
    return "OK"

app.run(host="0.0.0.0", port=5000)