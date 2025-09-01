from flask import Flask, request
import pandas as pd
for_excel = [["temperature","humidity","pressure"]]
app = Flask(__name__)

def to_excel():
    global for_excel
    
    df = pd.DataFrame(for_excel)
    df.to_csv("client_data.csv", mode="w", header=False)
    
    for_excel = [["temperature","humidity","pressure"]]
    

@app.route('/data', methods=['POST'])
def data():
    global for_excel
    data = request.get_json()
    temp = data["temperature"]
    hum  = data["humidity"]
    pres = data["pressure"]
    return "OK"

app.run(host="0.0.0.0", port=5000)