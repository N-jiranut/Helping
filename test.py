import pandas as pd

head = ["flex_raw_1", "flex_raw_2", "flex_raw_3", "flex_raw_4","label"]
# head = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
main = []

# for i in range(6):
#     for side in ["l", "r"]:
#         for label in head:
#             main.append(f"{side}_{label}_{i}")
for label in head:
    main.append(label)

df = pd.DataFrame(columns=main)
df.to_csv('data/prime.csv', index=False)

# for i in range(12):
#     for side in ["l", "r"]:
#         main.append(f"{side}_accel_x_{i}")
#         main.append(f"{side}_accel_y_{i}")
#         main.append(f"{side}_accel_z_{i}")
#         main.append(f"{side}_gyro_x_{i}")
#         main.append(f"{side}_gyro_y_{i}")
#         main.append(f"{side}_gyro_z_{i}")
# print(main)
# print(len(main))