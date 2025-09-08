import pandas as pd

head = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "temperature", "flex_raw_1", "flex_raw_2", "flex_raw_3", "flex_raw_4", "flex_raw_5"]
main = []

for i in range(12):
    for side in ["l", "r"]:
        for label in head:
            main.append(f"{side}_{label}_{i}")

df = pd.DataFrame(columns=main)
df.to_csv('data/main.csv', index=False)