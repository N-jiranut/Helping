import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("asl_translator/data/raw/RE_14-Aug.csv")
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values  


scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)  