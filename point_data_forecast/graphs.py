import matplotlib
import matplotlib.pyplot as plt
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

select = 6  # Select dataset
page = 2  # Select Set

file_path = datasets[select]

sheet = "Set_1" if page == 1 else "Set_2"
df = pd.read_excel(f"{main_path}/results/fixed/results_{file_path}_{sheet}.xlsx")
pred = df.set_index("Date")

"""# GRAPHS"""

matplotlib.rcParams["figure.figsize"] = (30, 10)

yrs = pred["2015":]  # Select year range

plt.plot(yrs["LE (W/m2)"])
plt.plot(yrs["1 day forecast"])  # Select forecast
