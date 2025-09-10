import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical

date="M9-5-2025"
name="unstable"

# Load your CSV
df = pd.read_csv("data/prime.csv", encoding="utf-8")

# Split features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values  

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
# print("Classes:", le.classes_)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

model = Sequential([
    Dense(8, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    # Dense(4, activation='relu'),
    # Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2000, batch_size=4, validation_data=(X_test, y_test), callbacks=[early_stop])

model.save(f"ML-model/{date}-{name}/model.h5")

with open(f"ML-model/{date}-{name}/text.txt", "w") as f:
    for label in le.classes_:
        f.write(str(label) + "\n")
        f.write("0")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"ML-model/{date}-{name}/keras_loss_plot.png")
plt.show()