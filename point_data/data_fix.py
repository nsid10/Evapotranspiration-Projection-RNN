import pandas as pd

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


def rescale(x, mode="minmax", a=None, b=None):
    return a + ((x - 0) * (b - a)) / (1 - 0)


def data_aug(select, page):
    file_path = datasets[select]

    sheet = "Set_1" if page == 1 else "Set_2"
    df = pd.read_excel(f"{main_path}/New_NLDAS/{file_path}.xlsx", sheet)
    df = df.set_index("Date")

    if select >= 0 and select <= 6:
        raw = df["2001":]
        top = "2017-01-01"
    else:
        raise Exception("Date range selection error")

    # raw

    pred = pd.read_excel(f"{main_path}/results/raw/predictions_multivar_raw_{file_path}_{sheet}.xlsx")
    del pred["Unnamed: 0"]
    pred["Date"] = raw.index
    pred = pred.set_index("Date")

    mini = df["LE (W/m2)"].min()
    maxo = df["LE (W/m2)"].max()
    print(mini, maxo)
    pred = rescale(pred, "custom", mini, maxo)

    pred = pd.concat((raw, pred), axis=1)

    # pred

    datelist = pd.date_range(start=top, periods=len(pred.columns) - 9).tolist()
    for day in datelist:
        pred = pred.append(pd.Series(name=day, dtype="float64"))

    for idx, col in enumerate(pred.columns):
        if idx < 8:
            continue
        pred[col] = pred[col].shift(idx - 8)

    # pred

    pred.to_excel(f"{main_path}/results/fixed/results_{file_path}_{sheet}.xlsx")
    print(f"Saved results_{file_path}_{sheet}.xlsx")


for s in range(7):
    for p in range(1, 3):
        data_aug(s, p)
