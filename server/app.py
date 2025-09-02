from flask import Flask, request
import pandas as pd
import numpy as np
app = Flask(__name__)
from tensorflow.keras.models import load_model
model = load_model(f"M9-2-2025-ta/model.h5")
with open(f"M9-2-2025-ta/text.txt", "r") as f:
    class_names = f.read().splitlines()

ldata = []
rdata = []
rc = False
lc = False

def predict():
    global ldata, rdata, rc, lc
    rc = False
    lc = False
    forpredict = ldata + rdata
    print(len(forpredict), "features")
    df = np.array([forpredict]).reshape(1, -1)
    pred = model.predict(df)
    index = np.argmax(pred)
    label = class_names[index]
    print(label)


@app.route('/right', methods=['POST'])
def rdata():
    global rdata, rc, lc
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
    if lc and rc:
        predict()
    return "OK"

@app.route('/left', methods=['POST'])
def ldata():
    global ldata, lc, rc
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
    if lc and rc:
        predict()
    return "OK"

app.run(host="0.0.0.0", port=5000)