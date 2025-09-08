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

def predict(side, data):
    global start, label, forpredict, ldata, rdata, lc, rc
    if start:
        if len(forpredict) < 288:
            if side == "l" and not lc:
                ldata = data
                lc = True
            elif side == "r" and not rc:
                rdata = data
                rc = True
            if lc and rc:
                forpredict.extend(ldata)
                forpredict.extend(rdata)
                lc = False
                rc = False
        else:
            df = pd.DataFrame([forpredict])
            df.to_csv('data/main.csv', mode="a", index=False, header=False)
            forpredict=[]
            start = False
            print(""*3)
            print("Data saved")
            print(""*3)

@app.route('/', methods=['POST', 'GET'])
def mainweb():
    return render_template('index.html', prediction=label)

@app.route('/btn', methods=['POST'])
def btn():
    global start
    print("Button pressed")
    if not start:
        start = True
    return "OK"

@app.route('/right', methods=['POST'])
def rget():
    global start
    if start:
        rc = True
        data = request.get_json()
        raccel_x = data["accel_x"]
        raccel_y = data["accel_y"]
        raccel_z = data["accel_z"]
        rgyro_x  = data["gyro_x"]
        rgyro_y  = data["gyro_y"]
        rgyro_z  = data["gyro_z"]
        rtemp = data["temperature"]
        rflex_1 = data["flex_raw_1"]
        rflex_2 = data["flex_raw_2"]
        rflex_3 = data["flex_raw_3"]
        rflex_4 = data["flex_raw_4"]
        rflex_5 = data["flex_raw_5"]
        rdata = [raccel_x, raccel_y, raccel_z, rgyro_x, rgyro_y, rgyro_z, rtemp, rflex_1, rflex_2, rflex_3, rflex_4, rflex_5]
        predict("r", rdata)
    return "OK"

@app.route('/left', methods=['POST'])
def lget():
    global ldata, lc, rc, start
    if start:
        lc = True
        data = request.get_json()
        laccel_x = data["accel_x"]
        laccel_y = data["accel_y"]
        laccel_z = data["accel_z"]
        lgyro_x  = data["gyro_x"]
        lgyro_y  = data["gyro_y"]
        lgyro_z  = data["gyro_z"]
        ltemp = data["temperature"]
        lflex_1 = data["flex_raw_1"]
        lflex_2 = data["flex_raw_2"]
        lflex_3 = data["flex_raw_3"]
        lflex_4 = data["flex_raw_4"]
        lflex_5 = data["flex_raw_5"]
        ldata = [laccel_x, laccel_y, laccel_z, lgyro_x, lgyro_y, lgyro_z, ltemp, lflex_1, lflex_2, lflex_3, lflex_4, lflex_5]
        predict("l", ldata)
    return "OK"

app.run(host="0.0.0.0", port=5000)