import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend as K

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.optimizers import Adam

global_seed = 42
np.random.seed(global_seed)

main_path = "/content/gdrive/My Drive/Projects/EV Projection LSTM"

datasets = [
    "TonziRanch_new",  # 0
    "US_ARM_new",  # 1
    "US_DPW_new",  # 2
    "US_LostCreek_new",  # 3
    "US_NE3_new",  # 4
    "US_NR1_new",  # 5
    "US_WalnutGulch_new",  # 6
]

select = 0
page = 1

file_path = datasets[select]


def normalize(x, mode="minmax", a=None, b=None):
    if mode == "minmax":
        # Min-max [0, 1]
        return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    if mode == "mean":
        # Mean normalization
        return (x - x.mean(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    if mode == "custom":
        # Custom range [a, b]
        return a + ((x - x.min(axis=0)) * (b - a)) / (x.max(axis=0) - x.min(axis=0))


sheet = "Set_1" if page == 1 else "Set_2"
df = pd.read_excel(f"{main_path}/New_NLDAS/{file_path}.xlsx", sheet)
df = df.set_index("Date")
data = normalize(df)


data.isnull().values.any()


def timeshift(data, timesteps=1, forecasts=1):
    df = pd.DataFrame(data)
    shifted = [df.shift(i) for i in range(timesteps, -forecasts, -1)]
    ts = pd.concat(shifted, axis=1)
    ts.columns = [f"t{1 - i}" for i in range(timesteps + 1, 1 - forecasts, -1)]
    ts.fillna(0, inplace=True)
    return ts


def timeSelect(data, select):
    if select == 0:
        data_train = ts_data["2001":"2015"]
        data_test = ts_data["2016":]
    else:
        raise Exception("error")
    return data_train, data_test


timesteps = 100
forecasts = 30

x_train = []
x_test = []

for feature in data:
    ts_data = timeshift(data[feature], timesteps, forecasts)
    data_train, data_test = timeSelect(ts_data, select)

    f_train = data_train.iloc[:, :-forecasts].to_numpy()
    f_test = data_test.iloc[:, :-forecasts].to_numpy()

    x_train.append(f_train.reshape(f_train.shape[0], f_train.shape[1], 1))
    x_test.append(f_test.reshape(f_test.shape[0], f_test.shape[1], 1))

    if feature == "LE (W/m2)":
        y_train = data_train.iloc[:, -forecasts:].to_numpy()
        y_test = data_test.iloc[:, -forecasts:].to_numpy()

x_train = np.concatenate(x_train, axis=2)
x_test = np.concatenate(x_test, axis=2)


def network(vector_dim, cast):
    input = Input(shape=vector_dim)

    rnn1 = LSTM(500, return_sequences=True)(input)
    drpx = Dropout(0.05)(rnn1)

    # for _ in range(2):
    #         rnnx = LSTM(500, return_sequences=True)(drpx)
    #         drpx = Dropout(0.05)(rnnx)

    rnn2 = LSTM(500)(drpx)
    drp2 = Dropout(0.05)(rnn2)

    output = Dense(cast)(drp2)

    model = tf.keras.Model(inputs=[input], outputs=[output])
    model.compile(optimizer=Adam(0.01), loss="mean_squared_error")

    return model


model = network((x_train.shape[1], x_train.shape[2]), y_train.shape[1])


checkpoint = ModelCheckpoint(
    filepath=f"{main_path}/models/checkpoint_{file_path}/", monitor="loss", save_best_only=True, save_freq="epoch"
)

reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.3, patience=25, verbose=1, cooldown=25, min_lr=1e-10)


def scheduler(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * tf.math.exp(-0.005)


lr_shceduler = LearningRateScheduler(scheduler, verbose=1)

K.set_value(model.optimizer.learning_rate, 0.01)

model.fit(x_train, y_train, epochs=500, batch_size=16, verbose=2, callbacks=[checkpoint, reduce_lr, lr_shceduler])

model = tf.keras.models.load_model(f"{main_path}/models/checkpoint_{file_path}")

x = np.concatenate((x_train, x_test), axis=0)
preds = model.predict(x)

formated_pred = pd.DataFrame(preds, columns=[f"{i + 1} day forecast" for i in range(preds.shape[1])])
formated_pred.to_excel(f"{main_path}/linear results raw/predictions_multivar_raw_{file_path}.xlsx")
