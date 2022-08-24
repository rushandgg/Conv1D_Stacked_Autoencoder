import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
import os


# Data Load
df = pd.read_csv('D:/Denoising/Autoencoder/Conv1D_SAE/Samsung.txt', sep = ',')


# Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scale_cols = ['Close']
scaled = scaler.fit_transform(df[["CLOSE"]])
df["Close"] = scaled


# Divide by Window Size
size = 56
total_stock_list = np.zeros((len(df)-size+1,size,1))
count = 0
for i in range(len(df)-size+1):
    stock_list = np.zeros(shape = size)
    end_idx = i+size
    dff = df[i:end_idx]
    dff.reset_index(inplace = True)
    if len(dff)>=size:
        for j in range(size):
            stock_list2 = np.zeros(shape = 1)
            stock_list2[0] = dff["Close"][j]
            stock_list[j] = stock_list2
            
            total_stock_list[count][j][0] = dff["Close"][j]            
        count += 1
    print("{} / {}".format(i+1, len(total_stock_list)))
    
    
# Split Train and Test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(total_stock_list, total_stock_list, test_size=0.2, random_state=42)


# Build Model
model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=64, kernel_size=8, padding="same", strides=2
        ),
        layers.Conv1D(
            filters=32, kernel_size=8, padding="same", strides=2
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=8, padding="same", strides=2
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=8, padding="same", strides=2
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=8, padding="same", strides=2
        ),
        layers.Conv1DTranspose(
            filters=64, kernel_size=8, padding="same", strides=2
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=8, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse")
model.summary()

plot_model(model, to_file='model_shapes.png', show_shapes=True)

# Train Model
history = model.fit(
    x_train,
    x_train,
    epochs=1000,
    batch_size=10,
    validation_split=0.1
)


# Check Loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()


# Check Smoothing, Compare with origin stock price
x_train_pred = model.predict(x_train)

fig, axs = plt.subplots(4,4, figsize=(13,10))
index = 0
for i in range(4):
    for j in range(4):
        axs[i,j].plot(x_train_pred[index], marker='.', c ='blue', label = 'Pred_Data')
        axs[i,j].plot(x_train[index], marker='.', c ='red', label = 'Ori_Data')
        index += 50
        #print(index)

#plt.savefig(str(size) + '.png')
plt.show()
