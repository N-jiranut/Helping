from flask import Flask, request, render_template
import pandas as pd
import numpy as np
app = Flask(__name__)

lc = False
rc = False
ldata = []
rdata = []
forpredict = []
label="Hello"
start = False
status = "Idle"

def predict(data):
    global start, label, forpredict, ldata, rdata, lc, rc, status
    if start:
        print("Predicting...")
        df = pd.DataFrame([data])
        df.to_csv('data/prime.csv', mode="a", index=False, header=False)
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
    if start:
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
        predict(ldata)
    return "OK"

app.run(host="0.0.0.0", port=5000)