import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image


main_path = "/content/gdrive/My Drive/Remote LSTM"
sub_paths = ["LostCreek_site_MI", "NE3_Site_NE", "TonziRanch_site_CA", "WalnutGulchWatershed_AZ"]


def display_images(my_images):
    row, col = 15, 6
    my_images = my_images[-1 * row * col :]
    my_images = np.reshape(my_images, (my_images.shape[0], my_images.shape[1], my_images.shape[2]))
    _, ax = plt.subplots(row, col, figsize=(col * 4, row * 4))
    for i, image in enumerate(my_images):
        ax[i // col, i % col].imshow(image, vmin=0, vmax=1)
        ax[i // col, i % col].title.set_text(f"{i + 1}")
        ax[i // col, i % col].axis("off")
    plt.show()
    plt.close()


def provide_info(data_path):
    box = os.listdir(data_path)

    potato = [date[23:27] for date in box]
    yr_list = list(set(potato))
    yr_list.sort()
    print("Data distribution\nYear\tCount")
    for yr in yr_list:
        cc = potato.count(yr)
        print(f"{yr}\t{cc}")
    print("\n\n\n")

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

    print(f"Max\t{data.max()}")
    print(f"Min\t{data.min()}")
    print(f"Mean\t{data.mean()}")
    print(f"99.9\t{np.percentile(data, 99.9)}")
    print(f"99.95\t{np.percentile(data, 99.95)}")
    print(f"99.99\t{np.percentile(data, 99.99)}")

    scale = np.percentile(data, 99.9)
    data = data / scale

    print(f"\nData shape = {data.shape}")

    return data


data = provide_info(f"{main_path}/MODIS Data/MODIS_LE_US_{data_path}")


display_images(data)
