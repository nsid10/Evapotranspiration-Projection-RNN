import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from PIL import Image
from tensorflow.keras.layers import BatchNormalization, Conv2D, ConvLSTM2D, Input


global_seed = 42
np.random.seed(global_seed)

main_path = ""
sub_paths = ["LostCreek_site_MI", "NE3_Site_NE", "TonziRanch_site_CA", "WalnutGulchWatershed_AZ"]

select = 0  # select dataset
data_path = sub_paths[select]

box = os.listdir(f"{main_path}/MODIS Data/MODIS_LE_US_{data_path}")


# loading data
data = []
threshold = 32000

for n in box:
    im = Image.open(f"{main_path}/MODIS Data/MODIS_LE_US_{data_path}/{n}")
    map = np.array(im)
    map = np.reshape(map, (map.shape[0], map.shape[1], 1))

    i = map > threshold
    map[i] = 0

    data.append(map)


data = np.array(data)
scale = scale = np.percentile(data, 99.9)
data = data / scale

h, w, c = data.shape[1], data.shape[2], data.shape[3]


frame_data = np.reshape(data, (data.shape[0], 1, h, w, c))


def shift_frames(data, shifts):
    """
    creates frame window for data sample
    """
    shifted_data = np.roll(data, 1, axis=0)
    shifted_data[0, :, :, :] = 0

    if shifts > 1:
        next = shift_frames(shifted_data, shifts - 1)
        shifted_data = np.concatenate((next, shifted_data), axis=1)

    return shifted_data


frames = 23
shifted_data = shift_frames(frame_data, frames)


for cutoff in range(len(box)):
    if box[cutoff][23:27] == "2019":
        break


if select in (0, 1, 2):
    start = 45
else:
    start = 23


y_train = data[start:cutoff]
y_test = data[cutoff:]

x_train = shifted_data[start:cutoff]
x_test = shifted_data[cutoff:]


print(y_train.shape)
print(y_test.shape)
print(x_train.shape)
print(x_test.shape)


def LE_map_model(input_dim):
    input = Input(shape=input_dim)

    cr1 = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(input)
    bn1 = BatchNormalization()(cr1)

    cr2 = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(bn1)
    bn2 = BatchNormalization()(cr2)

    cr3 = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(bn2)
    bn3 = BatchNormalization()(cr3)

    output = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(bn3)

    model = tf.keras.Model(inputs=[input], outputs=[output])
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


def LE_map_model2(input_dim):
    input = Input(shape=input_dim)

    cr1 = ConvLSTM2D(50, (3, 3), padding="same", return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(input)
    bn1 = BatchNormalization()(cr1)

    cr2 = ConvLSTM2D(50, (3, 3), padding="same", return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(bn1)
    bn2 = BatchNormalization()(cr2)

    cr3 = ConvLSTM2D(50, (3, 3), padding="same", return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(bn2)
    bn3 = BatchNormalization()(cr3)

    cr4 = ConvLSTM2D(50, (3, 3), padding="same", return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(bn3)
    bn4 = BatchNormalization()(cr4)

    output = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(bn4)

    model = tf.keras.Model(inputs=[input], outputs=[output])
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


# model training
model = LE_map_model2((frames, h, w, c))

model.summary(line_length=150)

model.fit(x_train, y_train, epochs=200, batch_size=16, verbose=2)

model.save(f"{main_path}/models/model_convlstm_{data_path}_{str(datetime.datetime.now())[:-7]}")


x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)


preds = model.predict(x)


def display_images(my_images):
    row, col = 16, 6
    my_images = my_images[-1 * row * col :]
    my_images = np.reshape(my_images, (my_images.shape[0], my_images.shape[1], my_images.shape[2]))
    _, ax = plt.subplots(row, col, figsize=(col * 4, row * 4))
    for i, image in enumerate(my_images):
        ax[i // col, i % col].imshow(image, vmin=0, vmax=1)
        ax[i // col, i % col].title.set_text(f"{i + 1}")
        ax[i // col, i % col].axis("off")
    plt.show()
    plt.close()


display_images(y)
display_images(preds)

preds = preds * scale
cap = box[start:]


for n, pic in enumerate(preds):
    pic = np.reshape(pic, (pic.shape[0], pic.shape[1]))
    im = Image.fromarray(pic)
    im.save(f"{main_path}/maps/{data_path}/LE_map_{cap[n][23:30]}.tif")
