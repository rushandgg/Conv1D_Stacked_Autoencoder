# Conv1D Stacked Autoencoder를 이용한 주가 디노이징

주가 예측 시 주가에 노이즈가 있어 그대로 넣는 것은 바람직하지 못하다.<br/>
보다 정확한 주가 예측을 위해 디노이징 연구를 진행한다.<br/>

Autoencoder는 데이터의 특징을 뽑아내 주는 장점이 있어 Autoencoder를 사용해 진행했다.
<br/><br/>

1. 데이터 로드 (삼성전자 주가, 2016-01-01 ~ 2021-12-31)

```c
df = pd.read_csv('D:/Denoising/Autoencoder/Samsung.txt', sep = ',')
```
<br/>

2. 정규화 작업

```c
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scale_cols = ['Close']
scaled = scaler.fit_transform(df[["CLOSE"]])
df["Close"] = scaled
```
<br/>

3. 주가를 Window Size 56으로 나눔
```c
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
```
<br/>

4. 모델 구성 (층 수, kernel size, activation에 따라 smoothing 정도를 확인할 예정)
```c
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
```
![model_shapes](https://user-images.githubusercontent.com/60992415/185845603-175530ea-29da-4a3e-ace4-75657fd8f0a8.png)
<br/><br/>

5. 모델 훈련
```c
history = model.fit(
    x_train,
    x_train,
    epochs=1000,
    batch_size=10,
    validation_split=0.1
)
```
<br/>

6. Loss 확인
```c
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
```
![Loss](https://user-images.githubusercontent.com/60992415/185844480-e6c60cc1-344e-4a2c-a076-3160ced45945.png)


7. 원래 주가와 비교하여 잘 smoothing 되었는지 확인한다.(빨간색이 원래 주가)
```c
x_train_pred = model.predict(x_train)

fig, axs = plt.subplots(4,4, figsize=(13,10))
index = 0
for i in range(4):
    for j in range(4):
        axs[i,j].plot(x_train_pred[index], marker='.', c ='blue', label = 'Pred_Data')
        axs[i,j].plot(x_train[index], marker='.', c ='red', label = 'Ori_Data')
        index += 50
plt.show()
```
![WindowSize56,Layers3,KernelSize8](https://user-images.githubusercontent.com/60992415/185844667-9c6f9c53-4d5e-489c-bb16-4c49a777d59e.png)



