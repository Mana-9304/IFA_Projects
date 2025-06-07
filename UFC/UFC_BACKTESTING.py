#!/usr/bin/env python
# coding: utf-8

# # Config


# In[1]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math
from datetime import datetime

import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    multilabel_confusion_matrix,
)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB, CategoricalNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from openpyxl import load_workbook
import itertools
from scraper_1 import script_runner

import warnings
warnings.filterwarnings("ignore")
# from catboost import CatBoostClassifier





def evaluate_model(y_true, y_pred):
    """
    :param y_true: ground truth values
    :param y_pred: predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # print("Accuracy", accuracy_score(y_true, y_pred))
    # print("precision", precision_score(y_true, y_pred, average = 'weighted'))
    # print("recall", recall_score(y_true, y_pred, average = 'weighted'))
    # print("F1", f1_score(y_true, y_pred, average = 'weighted'))
    # print("ROC_AUC ", roc_auc_score(y_true, y_pred, average = 'weighted'))

    report = classification_report(y_true, y_pred)
    print("Classification Report\n", report)

    cm = confusion_matrix(y_true, y_pred)
    # tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
    # display_label = np.unique(y_true)
    # cm_display = ConfusionMatrixDisplay(
    #     confusion_matrix=cm, display_labels=display_label
    # )
    # cm_display.plot()
    # plt.show()
    
def calculate_age_in_days(row):
    if pd.isna(row["DOB Month"]) or pd.isna(row["DOB Day"]) or pd.isna(row["DOB Year"]):
        return None
    birth_date = datetime(
        year=int(row["DOB Year"]), month=int(row["DOB Month"]), day=int(row["DOB Day"])
    )
    age_in_days = (row["date"] - birth_date).days
    return age_in_days


def calculate_elo(winner_elo, loser_elo, k=32):
    expected_win = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loss = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))
    new_winner_elo = winner_elo + k * (1 - expected_win)
    new_loser_elo = loser_elo + k * (0 - expected_loss)
    return new_winner_elo, new_loser_elo

def calculate_elo_v2(winner_elo, loser_elo, method, base_k=32):
    # Define K-factor multipliers based on method
    method_multiplier = {
        'SUB': 1.5,          # Submission
        'M-DEC': 1.2,        # Majority Decision
        'KO/TKO': 1.5,       # Knockout/Technical Knockout
        'U-DEC': 1.1,        # Unanimous Decision
        'S-DEC': 0.8,        # Split Decision
        'Overturned': 0,     # Overturned, no rating change
        'CNC': 0,            # No Contest, no rating change
        'DQ': 0.75           # Disqualification
    }

    # Get the multiplier based on the method of victory
    k = base_k * method_multiplier.get(method, 1)

    # Calculate the expected win/loss probabilities
    expected_win = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    
    # Update Elo ratings
    new_winner_elo = winner_elo + k * (1 - expected_win)
    new_loser_elo = loser_elo - k * expected_win
    return new_winner_elo, new_loser_elo

# In[2]


# Update historical data

historical_file_path = "G:/ONEDRIVE/Anmol Vivek Atharva/UFC/historical_data.csv"
new_hist_file_path = "C:/Users/Dell/OneDrive/Anmol Vivek Atharva/UFC/Results/TEST/Past_fight_data.csv"
future_file_path = "C:/Users/Dell/OneDrive/Anmol Vivek Atharva/UFC/Results/TEST/Future_fight_data.csv"

# script_runner()

hist = pd.read_csv(historical_file_path, encoding="latin1")
new_hist = pd.read_csv(new_hist_file_path, encoding="latin1")
future = pd.read_csv(future_file_path, encoding="latin1")

if isinstance(hist["Event Date"].iloc[-1],str):
    try:
        last_date_hist = datetime.strptime(hist["Event Date"].iloc[-1],"%d-%m-%Y")
    except:
        last_date_hist = datetime.strptime(hist["Event Date"].iloc[-1],"%Y-%m-%d")
    try:
        last_date_past = datetime.strptime(new_hist["Event Date"].iloc[-1],"%d/%m/%Y")
    except:
        last_date_past = datetime.strptime(new_hist["Event Date"].iloc[-1],"%Y/%m/%d")
    try:
        last_date_future = datetime.strptime(future["Event Date"].iloc[-1],"%d/%m/%Y")
    except:
        last_date_future = datetime.strptime(future["Event Date"].iloc[-1],"%Y/%m/%d")

print(hist.shape)
print(datetime.now(),hist["Event Date"].iloc[-1], hist['Date of Birth'])

# if last_date_past == last_date_hist:

#     hist['Event Date'] = pd.to_datetime(hist['Event Date'],dayfirst=True)
#     hist['Date of Birth'] = hist['Date of Birth'].astype(str)
#     # hist['Date of Birth'] = pd.to_datetime(hist['Date of Birth'],dayfirst=True)
#     hist['Date of Birth'] = pd.to_datetime(hist['Date of Birth'], errors='coerce', dayfirst=True)
#     new_hist['Event Date'] = pd.to_datetime(new_hist['Event Date'],dayfirst=True)
#     new_hist['Date of Birth'] = pd.to_datetime(new_hist['Date of Birth'],dayfirst=True)

#     new_hist = new_hist.sort_values(by=['Event Date', 'Fight ID']).reset_index(drop=True)


#     hist = hist[hist['Event Date']<last_date_past]

#     fight_id = len(hist)/2
#     for itr in range(0,len(new_hist)-1,2):
#         fight_id+=1
#         new_hist['Fight ID'].iloc[itr] = fight_id
#         new_hist['Fight ID'].iloc[itr+1] = fight_id

#     data: pd.DataFrame = pd.concat([hist, new_hist], axis=0).drop_duplicates()
#     data.sort_values('Fight ID',inplace=True)
#     data.reset_index()

# else:
#     hist['Event Date'] = pd.to_datetime(hist['Event Date'])
#     hist['Date of Birth'] = pd.to_datetime(hist['Date of Birth'])
#     data = hist



# if last_date_future >= last_date_hist:

#     future['Event Date'] = pd.to_datetime(future['Event Date'],dayfirst=True)
#     future['Date of Birth'] = pd.to_datetime(future['Date of Birth'],dayfirst=True)
    
#     data = data[data['Event Date']<last_date_future]
    
#     fight_id = len(data)/2
#     for itr in range(0,len(future)-1,2):
#         fight_id+=1
#         future['Fight ID'].iloc[itr] = fight_id
#         future['Fight ID'].iloc[itr+1] = fight_id

#     data = pd.concat([data, future], axis=0).drop_duplicates()
#     data.sort_values('Fight ID',inplace=True)
#     data.reset_index()

#     data.to_csv("historical_data.csv",index=False)
# else:
#     hist['Event Date'] = pd.to_datetime(hist['Event Date'])
#     hist['Date of Birth'] = pd.to_datetime(hist['Date of Birth'])
#     data = hist


hist.rename(columns={'date': 'Event Date'}, inplace=True)
hist['Date of Birth'] = hist['Date of Birth'].apply(lambda x: "-".join(str(x).split("-")[::-1]) if not x=='' else '')
hist['Event Date'] = pd.to_datetime(hist['Event Date'])
hist['Date of Birth'] = pd.to_datetime(hist['Date of Birth'])
data = hist

data['Event Date Month'] = data['Event Date'].dt.month
data['Event Date Day'] = data['Event Date'].dt.day
data['Event Date Year'] = data['Event Date'].dt.year
data['odds'] = data['odds'].apply(lambda x: x if x>0 else round(-10000/x,0))

data['DOB Month'] = data['Date of Birth'].dt.month
data['DOB Day'] = data['Date of Birth'].dt.day
data['DOB Year'] = data['Date of Birth'].dt.year

data.sort_values('Fight ID')
data = data.drop(['Event Date','Date of Birth'], axis=1)

print(data.shape)

data.tail(6)

# In[3]
    
# Create datetime feature to sort by event timeline
data["date"] = (
    data["Event Date Day"].astype(int).astype(str)
    + "-"
    + data["Event Date Month"].astype(int).astype(str)
    + "-"
    + data["Event Date Year"].astype(int).astype(str)
)
data["date"] = pd.to_datetime(data["date"], format="%d-%m-%Y")

# Concat full name fighter
data["Winner Full Name"] = (
    data["Winner First Name"].fillna("").str.upper() + " " + data["Winner Last Name"].fillna("").str.upper()
)
data["Fighter Full Name"] = (
    data["Fighter First Name"].fillna("").str.upper() + " " + data["Fighter Last Name"].fillna("").str.upper()
)

data["Age (in days)"] = data.apply(calculate_age_in_days, axis=1)
data["Height"] = data["Height Feet"] * 12 + data["Height Inches"]

def time_converter(x):
    try:
        a = float(x)  # Ensure x is converted to float
    except ValueError:
        return np.nan  # Return NaN if conversion fails
    if np.isnan(a):
        return 0
    elif a < 1:
        return a * 100
    return np.floor(a) * 60 + (a * 100) % 100

data["Ground and Cage Control Time"] = data["Ground and Cage Control Time"].apply(lambda x:time_converter(x))
data["Winning Time"] = data["Winning Time"].apply(lambda x:time_converter(x))

fillna_features = [
    "Winning Time",
    "Ground and Cage Control Time",
    "Knockdown Total",
    "Takedown Total Attempted",
    "Takedown Total Landed",
    "Significant Strike Total Attempted",
    "Significant Strike Total Landed",
    "Significant Strike Head Attempted",
    "Significant Strike Head Landed",
    "Significant Strike Body Attempted",
    "Significant Strike Body Landed",
    "Significant Strike Leg Attempted",
    "Significant Strike Leg Landed",
    "Significant Strike Clinch Attempted",
    "Significant Strike Clinch Landed",
    "Significant Strike Ground Attempted",
    "Significant Strike Ground Landed",
]
data[fillna_features] = data[fillna_features].fillna(0)

data = (
    data.drop_duplicates(subset = ["date", "Winner Full Name", "Fighter Full Name"])
    .sort_values(["date", "Fight ID", "Winner Full Name", "Fighter Full Name"])
    .reset_index(drop=True)
)


# # Feature Engineer




data["NumberOf_Fight"] = data.groupby(["Fighter Full Name"], as_index=False)[
    "Fight ID"
].transform(lambda x: x.shift(1).rolling(100, min_periods=1).count())
data["IS_WIN"] = np.where(data["Fighter Full Name"] == data["Winner Full Name"], 1, 0)

data["NumberOf_WIN"] = data.groupby(["Fighter Full Name"])["IS_WIN"].transform(
    lambda x: x.shift(1).rolling(window=100, min_periods=1).sum()
)
data["NumberOf_LOSE"] = data["NumberOf_Fight"] - data["NumberOf_WIN"]
data[["NumberOf_Fight", "NumberOf_WIN"]] = data.groupby(["Fighter Full Name"])[
    ["NumberOf_Fight", "NumberOf_WIN"]
].transform(lambda x: x.ffill())
data[["NumberOf_Fight", "NumberOf_WIN", "NumberOf_LOSE"]] = data[
    ["NumberOf_Fight", "NumberOf_WIN", "NumberOf_LOSE"]
].fillna(0)
data["WIN_RATE"] = pd.to_numeric(
    data["NumberOf_WIN"] / data["NumberOf_Fight"], errors="coerce"
).fillna(0)
data["WIN_RATE"] = data["WIN_RATE"].astype(float)
data = data.drop(["IS_WIN"], axis=1)


data["ELO_fighter"] = 1500
data["ELO_winner"] = 1500
for index, row in data.iterrows():
    if (row["Fighter Full Name"] != row["Winner Full Name"]) & (
        row["Winner Full Name"] != "Draw Draw"
    ):
        jump_step = 0
        try: 
            if (
                (data.loc[index, "Winner Full Name"] == data.loc[index + 1, "Fighter Full Name"])
                & (data.loc[index, "Fight ID"] == data.loc[index + 1, "Fight ID"])
            ):
                jump_step = 1
        except: pass
            
        winner_name = row["Winner Full Name"]
        loser_name = row["Fighter Full Name"]
        new_winner_elo, new_loser_elo = calculate_elo(
            data.loc[index, "ELO_winner"], data.loc[index, "ELO_fighter"]
        )
        data.loc[
            (data["Fighter Full Name"] == winner_name) & (data.index > index+jump_step),
            "ELO_fighter",
        ] = new_winner_elo
        data.loc[
            (data["Winner Full Name"] == winner_name) & (data.index > index+jump_step),
            "ELO_winner",
        ] = new_winner_elo

        data.loc[
            (data["Fighter Full Name"] == loser_name) & (data.index > index+jump_step),
            "ELO_fighter",
        ] = new_loser_elo
        data.loc[
            (data["Winner Full Name"] == loser_name) & (data.index > index+jump_step),
            "ELO_winner",
        ] = new_loser_elo



data["new_ELO_fighter"] = 1500
data["new_ELO_winner"] = 1500

for index, row in data[:-1].iterrows():
    if (row["Fighter Full Name"] != row["Winner Full Name"]) & (
        row["Winner Full Name"] != "Draw Draw"
    ):
        jump_step = 0
        if (
            (data.loc[index, "Winner Full Name"] == data.loc[index + 1, "Fighter Full Name"])
            & (data.loc[index, "Fight ID"] == data.loc[index + 1, "Fight ID"])
        ):
            jump_step = 1
            
        winner_name = row["Winner Full Name"]
        loser_name = row["Fighter Full Name"]
        method = row["Winning Method"]  # Assuming you have a 'Win Method' column

        new_winner_elo, new_loser_elo = calculate_elo_v2(
            data.loc[index, "new_ELO_winner"], data.loc[index, "new_ELO_fighter"], method
        )

        # Update Elo ratings for winner and loser
        data.loc[
            (data["Fighter Full Name"] == winner_name) & (data.index > index+jump_step),
            "new_ELO_fighter",
        ] = new_winner_elo
        data.loc[
            (data["Winner Full Name"] == winner_name) & (data.index > index+jump_step),
            "new_ELO_winner",
        ] = new_winner_elo

        data.loc[
            (data["Fighter Full Name"] == loser_name) & (data.index > index+jump_step),
            "new_ELO_fighter",
        ] = new_loser_elo
        data.loc[
            (data["Winner Full Name"] == loser_name) & (data.index > index+jump_step),
            "new_ELO_winner",
        ] = new_loser_elo


# In[4]

data['original_index'] = data.index

# Step 1: Sort the data by Fighter Full Name and date to ensure calculations are done chronologically within each fighter
data = data.sort_values(by=['Fighter Full Name', 'date'])

# Step 2: Create the is_win column (1 if the fighter won, 0 if lost)
data['is_win'] = (data["Fighter Full Name"] == data["Winner Full Name"]).astype(int)

# Step 3: Shift the is_win and Knockdown Total columns to use previous fight's data
data['prev_is_win'] = data.groupby('Fighter Full Name')['is_win'].shift(1)

# data['prev_knockdown_total'] = data.groupby('Fighter Full Name')['Knockdown Total'].shift(1)

# # Step 4: Calculate total knockdowns for wins (only for previous winning fights)
# data['total_win_knockdown'] = data.groupby("Fighter Full Name")["prev_knockdown_total"].transform(
#     lambda x: x.where(data['prev_is_win'] == 1).cumsum()
# )

# # Step 5: Calculate total knockdowns for losses (only for previous losing fights)
# data['total_lose_knockdown'] = data.groupby("Fighter Full Name")["prev_knockdown_total"].transform(
#     lambda x: x.where(data['prev_is_win'] == 0).cumsum()
# )

# # Step 6: Calculate win average knockdown total and apply forward fill for missing values
# data['WIN_AVG_Knockdown Total'] = data.groupby("Fighter Full Name")["total_win_knockdown"].transform(
#     lambda x: x / data['NumberOf_WIN']
# ).ffill()

# # Step 7: Calculate lose average knockdown total and apply forward fill for missing values
# data['LOSE_AVG_Knockdown Total'] = data.groupby("Fighter Full Name")["total_lose_knockdown"].transform(
#     lambda x: x / data['NumberOf_LOSE']
# ).ffill()


# data.drop(['is_win','prev_is_win','prev_knockdown_total','total_win_knockdown','total_lose_knockdown'],axis=1,inplace=True)
# data = data.sort_values(by='original_index').drop(columns=['original_index'])
fighter_list = data['Fighter Full Name'].unique()
ddata = pd.DataFrame()
new_feat_list = []
for k in fighter_list:
    # if k=='CODY DURDEN':
    #     print('HI')
    data_fighter = data[data['Fighter Full Name']==k]
    data_fighter_odds = data_fighter['odds']
    # print(k)
    for i in fillna_features:
        col_name = i
        prev_col_name = "prev_"+i
        total_win_col_name = "total_win_"+i
        total_lose_col_name = "total_lose_"+i
        win_avg_col_name = "win_avg_"+i
        lose_avg_col_name = "lose_avg_"+i

        data_fighter[prev_col_name] = data_fighter.groupby('Fighter Full Name')[col_name].shift(1)

        # Step 4: Calculate total knockdowns for wins (only for previous winning fights)
        data_fighter[total_win_col_name] = data_fighter.groupby("Fighter Full Name")[prev_col_name].transform(
            lambda x: x.where(data_fighter['prev_is_win'] == 1).cumsum()
        )

        # Step 5: Calculate total knockdowns for losses (only for previous losing fights)
        data_fighter[total_lose_col_name] = data_fighter.groupby("Fighter Full Name")[prev_col_name].transform(
            lambda x: x.where(data_fighter['prev_is_win'] == 0).cumsum()
        )

        # Step 6: Calculate win average knockdown total and apply forward fill for missing values
        data_fighter[win_avg_col_name] = data_fighter.groupby("Fighter Full Name")[total_win_col_name].transform(
            lambda x: x / data_fighter['NumberOf_WIN']
        ).ffill()

        # Step 7: Calculate lose average knockdown total and apply forward fill for missing values
        data_fighter[lose_avg_col_name] = data_fighter.groupby("Fighter Full Name")[total_lose_col_name].transform(
            lambda x: x / data_fighter['NumberOf_LOSE']
        ).ffill()
        data_fighter.drop([prev_col_name,total_win_col_name,total_lose_col_name],axis=1,inplace=True)
        
        if win_avg_col_name in new_feat_list:
            continue
        else:
            new_feat_list.append(win_avg_col_name)
            new_feat_list.append(lose_avg_col_name)

    data_fighter['odds'] = data_fighter_odds
    ddata = ddata._append(data_fighter,ignore_index=True)        

new_feat_list.append("odds")
ddata = ddata.fillna(0)
ddata.drop(['is_win','prev_is_win'],axis=1,inplace=True)
# ddata['odds'] = data['odds']
ddata = ddata.sort_values(by='original_index').drop(columns=['original_index'])


# In[5]


# data[data["Fighter Full Name"] == "NEIL MAGNY"][
#     [
#         "date",
#         "Fight ID",
#         "Fighter Full Name",
#         "Winner Full Name",
#         "NumberOf_Fight",
#         "NumberOf_WIN",
#         "NumberOf_LOSE",
#         "WIN_RATE",
#         "ELO_fighter",
#         "new_ELO_fighter",
#         "WIN_AVG_Knockdown Total",
#         "LOSE_AVG_Knockdown Total",
#     ]
# ].head()





# transform_col = [
#     'NumberOf_Fight', 'NumberOf_WIN', 'NumberOf_LOSE', 'WIN_RATE', 'ELO_fighter', "new_ELO_fighter",
#     'WIN_AVG_Knockdown Total', 'LOSE_AVG_Knockdown Total'    
# ]
transform_col = [
    'NumberOf_WIN', 'NumberOf_LOSE',
    #   'WIN_RATE',
    # 'WIN_AVG_Knockdown Total', 'LOSE_AVG_Knockdown Total'    
]
####################################################################################################################
# for lag_idx in range(1,4):
# #     new_col = [col + f"_shift_{lag_idx+1}" for col in transform_col]
# #     data[new_col] = data.groupby(["Fighter Full Name"])[transform_col].shift(lag_idx)
# # lag_idx = 2
#     new_col = [col + f"_shift_{lag_idx+1}" for col in transform_col]
#     data[new_col] = data.groupby(["Fighter Full Name"])[transform_col].shift(lag_idx)

#     # ['NumberOf_WIN', 'NumberOf_LOSE', 'WIN_RATE']
#     data["NumberOf_Fight_temp"] = data.groupby(["Fighter Full Name"], as_index=False)[
#         "Fight ID"
#     ].transform(lambda x: x.shift(1).rolling(window = lag_idx + 1, min_periods=1).count())
#     data["IS_WIN_temp"] = np.where(data["Fighter Full Name"] == data["Winner Full Name"], 1, 0)
#     data[f"NumberOf_WIN_shift_{lag_idx+1}"] = data.groupby(["Fighter Full Name"])["IS_WIN_temp"].transform(
#         lambda x: x.shift(1).rolling(window=lag_idx+1, min_periods=1).sum()
#     )
#     data[f"NumberOf_LOSE_shift_{lag_idx+1}"] = data["NumberOf_Fight_temp"] - data[f"NumberOf_WIN_shift_{lag_idx+1}"]
#     data[["NumberOf_Fight_temp", f"NumberOf_WIN_shift_{lag_idx+1}"]] = data.groupby(["Fighter Full Name"])[
#         ["NumberOf_Fight_temp", f"NumberOf_WIN_shift_{lag_idx+1}"]
#     ].transform(lambda x: x.ffill())
#     data[["NumberOf_Fight_temp", f"NumberOf_WIN_shift_{lag_idx+1}", f"NumberOf_LOSE_shift_{lag_idx+1}"]] = data[
#         ["NumberOf_Fight_temp", f"NumberOf_WIN_shift_{lag_idx+1}", f"NumberOf_LOSE_shift_{lag_idx+1}"]
#     ].fillna(0)
#     data[f"WIN_RATE_shift_{lag_idx+1}"] = pd.to_numeric(
#         data[f"NumberOf_WIN_shift_{lag_idx+1}"] / data["NumberOf_Fight_temp"], errors="coerce"
#     ).fillna(0)
#     data[f"WIN_RATE_shift_{lag_idx+1}"] = data[f"WIN_RATE_shift_{lag_idx+1}"].astype(float)
#     data = data.drop(["IS_WIN_temp","NumberOf_Fight_temp"], axis=1)

# # prepare data




features = []
outcomes = []
ddata = ddata.sort_values(["date", "Fight ID"]).reset_index(drop=True)

# Iterate over the data in steps of 2 rows (one for Fighter A, one for Fighter B)
for i in range(0, len(ddata), 2):
    if i + 1 >= len(ddata):
        break
    row_a = ddata.iloc[i]
    row_b = ddata.iloc[i + 1]

    # Randomly decide who is Fighter A and who is Fighter B to avoid bias
    if np.random.rand() > 0.5:
        row_a, row_b = row_b, row_a

    feature_diff = {
        "fight_id": row_a["Fight ID"],  # Adding Fight ID to feature_diff
        "date": row_a["date"],
        "fighter_a_name": row_a["Fighter Full Name"],
        "fighter_b_name": row_b["Fighter Full Name"],
    }
    features.append(feature_diff)

    if row_a["Winner Full Name"] == row_a["Fighter Full Name"]:
        outcomes.append(1)
    else:
        outcomes.append(0)

    row_a, row_b = row_b, row_a
    feature_diff = {
        "fight_id": row_a["Fight ID"],  # Adding Fight ID to feature_diff
        "date": row_a["date"],
        "fighter_a_name": row_a["Fighter Full Name"],
        "fighter_b_name": row_b["Fighter Full Name"],
    }
    features.append(feature_diff)

    if row_a["Winner Full Name"] == row_a["Fighter Full Name"]:
        outcomes.append(1)
    else:
        outcomes.append(0)

df = pd.concat([pd.DataFrame(features), pd.Series(outcomes)], axis=1)
df.columns = ["fight_id", "date", "fighter_x_name", "fighter_y_name", "fighter_x_win"]
df = df.sort_values(["date", "fight_id"]).reset_index(drop=True)

df





# merge biographical characteristics of fighter and historical dynamic features in each fight for training set
data_bio_feat_list = [
        "Fight ID",
        "date",
        "Fighter Full Name",
        "Age (in days)",
        "Height",
        "ELO_fighter",
        "new_ELO_fighter",
        "NumberOf_Fight",
        "NumberOf_WIN",
        "NumberOf_LOSE",
        "WIN_RATE",
        "Height Feet",
        "Height Inches",
        "Weight Pounds",
        "Reach Inches",
        "Stance",
        "DOB Month",
        "DOB Day",
        "DOB Year",
        ] + new_feat_list 
# + ['NumberOf_WIN_shift_2', 'NumberOf_LOSE_shift_2', 'WIN_RATE_shift_2', 
#         # 'ELO_fighter_shift_2', "new_ELO_fighter_shift_2",
#         # 'WIN_AVG_Knockdown Total_shift_2', 'LOSE_AVG_Knockdown Total_shift_2',
#         # 'NumberOf_Fight_shift_3',
#         'NumberOf_WIN_shift_3', 'NumberOf_LOSE_shift_3', 'WIN_RATE_shift_3', 
#         # 'ELO_fighter_shift_3',"new_ELO_fighter_shift_3",
#         # 'WIN_AVG_Knockdown Total_shift_3', 'LOSE_AVG_Knockdown Total_shift_3',
#         # 'NumberOf_Fight_shift_4', 
#         'NumberOf_WIN_shift_4', 'NumberOf_LOSE_shift_4', 'WIN_RATE_shift_4',
#         # 'ELO_fighter_shift_4',"new_ELO_fighter_shift_4",
#         # 'WIN_AVG_Knockdown Total_shift_4', 'LOSE_AVG_Knockdown Total_shift_4',
#         ]
data_bio = ddata[data_bio_feat_list].drop_duplicates()

df = df.merge(
    data_bio,
    left_on=["fight_id", "date", "fighter_x_name"],
    right_on=["Fight ID", "date", "Fighter Full Name"],
    how="left",
)
df = df.merge(
    data_bio,
    left_on=["fight_id", "date", "fighter_y_name"],
    right_on=["Fight ID", "date", "Fighter Full Name"],
    how="left",
)
print(df.shape)
df.tail()


# In[6]


# exclude "fighter_id", "date" features and "Fighter Full Name" as well to avoid duplicate fighter name with "fighter_x/y_name"
exclude_feature = [
    feature
    for feature in df.columns
    # if "id" in feature.lower() or "fighter full name" in feature.lower()
    if "fighter full name" in feature.lower()
]
print(exclude_feature)
df = df.drop(
    [
        *exclude_feature,
        # "date"
    ],
    axis=1,
)

# astype categorical features for fitting ensemble model
categorical_cols = [feature for feature in df.columns if df[feature].dtype == "O"]
df[categorical_cols] = df[categorical_cols].astype("category")
categorical_cols



fixed_col = [
    "Fighter Full Name",'Height Feet', 'Height Inches',
    'Stance', "date","Fight ID",
]

dynamic_col = data_bio.columns.drop(fixed_col)

for col in dynamic_col:
    df[f"{col}_diff"] = df[f"{col}_x"] - df[f"{col}_y"]
    
dynamic_col


# Correlation calculation with fighter_x_win
numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
fighter_x_win_correlation = numeric_df.corrwith(df['fighter_x_win']).sort_values(ascending=False)
print("Correlation with fighter_x_win:")
print(fighter_x_win_correlation)


data_with_leakage = df.copy()

data_without_leakage = df.drop(['fighter_x_win'],axis=1)

file_path = "G:/ONEDRIVE/Anmol Vivek Atharva/UFC/Input_files/backtest_prediction_202501_v3_22032025.xlsx"

# In[]

df.to_csv('G:/ONEDRIVE/Anmol Vivek Atharva/UFC/Input_files/Cleaned and manipulated UFC data_202501_v2_22032025.csv',index=False)

# In[7]
# df = pd.read_csv('G:/ONEDRIVE/Anmol Vivek Atharva/UFC/Input_files/Cleaned and manipulated UFC data_202501_v2.csv')
# data_with_leakage = df.copy()

# data_without_leakage = df.drop(['fighter_x_win'],axis=1)

# file_path = "G:/ONEDRIVE/Anmol Vivek Atharva/UFC/Input_files/backtest_prediction_202501_v3.xlsx"
try:
    os.remove(file_path)
except:
    print("File not found")

model_trained = 0

future_dates = df[df['date'] > '2022-06-04']['date'].unique()
# future_dates = df[df['date'] > '2024-05-18']['date'].unique() # THIS LINE IS ADDED ON 22032025


for future_date in future_dates:

    print(future_date)

    train_df_with_leakage = data_with_leakage[data_with_leakage['date']<future_date]

    train_df = train_df_with_leakage.copy()

    new_future = data_without_leakage[data_without_leakage['date']==future_date]
    
    fill_0_cols = ['NumberOf_Fight', 'NumberOf_WIN', 'NumberOf_LOSE', 'WIN_RATE']
    # fill_0_cols = [
    #         'NumberOf_Fight', 'NumberOf_WIN', 'NumberOf_LOSE', 'WIN_RATE',
    #         # 'NumberOf_Fight_shift_2', 
    #         'NumberOf_WIN_shift_2', 'NumberOf_LOSE_shift_2', 'WIN_RATE_shift_2',
    #         # 'NumberOf_Fight_shift_3',
    #         'NumberOf_WIN_shift_3', 'NumberOf_LOSE_shift_3', 'WIN_RATE_shift_3',
    #         # 'NumberOf_Fight_shift_4',
    #         'NumberOf_WIN_shift_4', 'NumberOf_LOSE_shift_4', 'WIN_RATE_shift_4',
    # ]
    fill_0_cols = list(''.join(e) for e in itertools.product(fill_0_cols, ["_x","_y"]))
    train_df[fill_0_cols] = train_df[fill_0_cols].fillna(0)

    elo_cols = [
            "ELO_fighter",
            # "ELO_fighter_shift_2","ELO_fighter_shift_3","ELO_fighter_shift_4",
            "new_ELO_fighter"
            # ,"new_ELO_fighter_shift_2","new_ELO_fighter_shift_3","new_ELO_fighter_shift_4"
    ]
    elo_cols = list(''.join(e) for e in itertools.product(elo_cols, ["_x","_y"]))
    train_df[elo_cols] = train_df[elo_cols].fillna(1500)

    bio_cols = [
            'Age (in days)', 'Height','Height Feet', 'Height Inches', 'Weight Pounds', 'Reach Inches',"Stance"] + new_feat_list 
                    # 'WIN_AVG_Knockdown Total', 'LOSE_AVG_Knockdown Total',
            # 'WIN_AVG_Knockdown Total_shift_2', 'LOSE_AVG_Knockdown Total_shift_2',
            # 'WIN_AVG_Knockdown Total_shift_3', 'LOSE_AVG_Knockdown Total_shift_3',
            # 'WIN_AVG_Knockdown Total_shift_4', 'LOSE_AVG_Knockdown Total_shift_4',
    

    for col in bio_cols:
            if train_df[f"{col}_x"].dtype == "category":
                    mode_category = train_df[f"{col}_x"].mode()[0]
                    train_df[[f"{col}_x",f"{col}_y"]] = train_df[[f"{col}_x",f"{col}_y"]].fillna(mode_category)
            else:
                    mean_imputation = train_df[[f"{col}_x",f"{col}_y"]].mean().mean()
                    train_df[[f"{col}_x",f"{col}_y"]] = train_df[[f"{col}_x",f"{col}_y"]].fillna(mean_imputation)
                    
    train_df.shape

    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    import lightgbm as lgb


    model = LGBMClassifier()

    if model_trained >= 0:
        # n_HP_points_to_test = 100
        # param_grid = {
        # 'num_leaves': [5, 20, 31],
        # 'learning_rate': [0.05, 0.1, 0.2],
        # 'n_estimators': [50, 100, 150]
        # }
        # model = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)
        # gs = RandomizedSearchCV(
        # estimator=model, param_distributions=param_grid, 
        # n_iter=n_HP_points_to_test,
        # scoring='roc_auc',
        # cv=3,
        # refit=True,
        # random_state=314,
        # verbose=True)
        model.fit(train_df.drop(["fight_id","Fight ID_x","Fight ID_y","fighter_x_win", "date",'fighter_x_name','fighter_y_name'], axis=1), train_df["fighter_x_win"])

        evaluate_model(train_df["fighter_x_win"],model.predict(train_df.drop(["fight_id","Fight ID_x","Fight ID_y","fighter_x_win", "date",'fighter_x_name','fighter_y_name'], axis=1)))
        model_trained = 1

    y_pred = model.predict(new_future.drop(["fight_id","Fight ID_x","Fight ID_y","date",'fighter_x_name','fighter_y_name'],axis=1))
    y_proba_all = model.predict_proba(new_future.drop(["fight_id","Fight ID_x","Fight ID_y","date",'fighter_x_name','fighter_y_name'],axis=1))[:, 1]

    new_future["x_win_predicted"]  = y_pred
    new_future["probability_x_win"] = y_proba_all

    new_future["y_win_predicted"]  = 1-y_pred
    new_future["probability_y_win"] = 1-y_proba_all
    
    # new_future['probability_x_win'] = new_future['probability_x_win'].apply(lambda x: x if x > 0.5 else 1 - x)
    
    # y_pred

    new_future['Predicted Winner'] = new_future.apply(lambda row: row['fighter_x_name'] if row['x_win_predicted'] == 1 else row['fighter_y_name'], axis=1)

    # new_future.head(4)


    # Check if the file already exists


    if os.path.exists(file_path):
        # Load the existing CSV into a DataFrame
        existing_df = pd.read_csv(file_path)
        # Append the new data (future_test DataFrame)       
        updated_df = pd.concat([existing_df, new_future], ignore_index=True)
        # Write the updated DataFrame back to the CSV file
        updated_df.to_csv(file_path, index=False)
    else:
        # If the file doesn't exist, write the future_test DataFrame to a new CSV file
        new_future.to_csv(file_path, index=False)

    # new_df = pd.concat([new_df, future_test])

print("File Saved Successfully")
# trail_df.drop(['index'], axis=1, inplace=True)
# trail_df.reset_index(drop=True, inplace=True)
# %%


actual_winner_df = df.copy()

actual_winner_df['actual_winner'] = np.where(actual_winner_df['date']<pd.to_datetime(datetime.now().date()),
                                             np.where(actual_winner_df['fighter_x_win'] == 1, 
                                                      actual_winner_df['fighter_x_name'], 
                                                      actual_winner_df['fighter_y_name']),np.nan)

trail_df = pd.merge(updated_df, actual_winner_df[["fight_id","actual_winner"]], on="fight_id", how="left")

trail_df.drop_duplicates(inplace=True)

trail_df['x_win_actual'] = trail_df.apply(lambda row: 1 if row['fighter_x_name']==row['actual_winner'] else 0, axis=1)
trail_df['y_win_actual'] = 1 - trail_df['x_win_actual']

trail_df["Accuracy"] = np.where(trail_df["actual_winner"] == trail_df["Predicted Winner"], 1, 0)

historical_data = pd.read_csv(historical_file_path)
# historical_data['fighter_full_name'] = historical_data['Fighter First Name'].str.upper() + ' ' + historical_data['Fighter Last Name'].str.upper()
# historical_data['fighter_full_name'] = historical_data['fighter_full_name'].str.strip()
historical_data['fighter_full_name'] = historical_data['Fighter First Name'].str.upper()
historical_data['fighter_full_name'] += ' '+historical_data['Fighter Last Name'].fillna('').str.upper()
historical_data['fighter_full_name'] = historical_data['fighter_full_name'].str.strip()

historical_data = historical_data[['Fight ID','fighter_full_name','odds']]
trail_df['fighter_x_name'] = trail_df['fighter_x_name'].str.strip()
trail_df['fighter_y_name'] = trail_df['fighter_y_name'].str.strip()

# First merge for fighter_x
trail_df = trail_df.merge(
    historical_data, 
    left_on=['Fight ID_x', 'fighter_x_name'], 
    right_on=['Fight ID', 'fighter_full_name'], 
    how='left'
)

# Rename columns to avoid conflict in second merge
trail_df.rename(columns={'odds': 'fighter_x_odds'}, inplace=True)
trail_df.drop(columns=['Fight ID', 'fighter_full_name'], inplace=True, errors='ignore')

# Second merge for fighter_y
trail_df = trail_df.merge(
    historical_data, 
    left_on=['Fight ID_y', 'fighter_y_name'], 
    right_on=['Fight ID', 'fighter_full_name'], 
    how='left'
)

# Rename and clean up
trail_df.rename(columns={'odds': 'fighter_y_odds'}, inplace=True)
trail_df.drop(columns=['Fight ID', 'fighter_full_name'], inplace=True, errors='ignore')

trail_df['Edge Above X'] = trail_df['probability_x_win'].apply(lambda x: -(x/(1-x))*100 if x>0.5 else ((1-x)/x)*100)
trail_df['Edge Above Y'] = trail_df['probability_y_win'].apply(lambda x: -(x/(1-x))*100 if x>0.5 else ((1-x)/x)*100)

trail_df['edge_x'] = trail_df['fighter_x_odds']-trail_df['Edge Above X']
trail_df['edge_y'] = trail_df['fighter_y_odds']-trail_df['Edge Above Y']


# trail_df.reset_index(drop=True, inplace=True)
trail_df.reset_index(inplace=True)
trail_df.drop(['index'], axis=1, inplace=True)
trail_df.reset_index(inplace=True)

def filter(row):
    # print(row['index'])
    if row['index']%2==0:
        if trail_df.loc[row['index'],'Predicted Winner'] == trail_df.loc[row['index']+1,'Predicted Winner']:
            value = trail_df.loc[row['index'],'Predicted Winner']
        else:
            value = np.nan
    else:
        if trail_df.loc[row['index']-1,'Predicted Winner'] == trail_df.loc[row['index'],'Predicted Winner']:
            value = trail_df.loc[row['index'],'Predicted Winner']
        else:
            value = np.nan
    return value


trail_df['filter'] = trail_df.apply(lambda row: filter(row), axis=1)
trail_df = trail_df[~trail_df['filter'].isna()]

trail_df['Bet_X'] = trail_df['fighter_x_odds'].apply(lambda x: x if x>0 else -10000/x)
trail_df['Bet_Y'] = trail_df['fighter_y_odds'].apply(lambda x: x if x>0 else -10000/x)

trail_df['profit_x'] = trail_df.apply(lambda row: row['Bet_X'] if row['x_win_actual']==1 else -100, axis=1)
trail_df['profit_y'] = trail_df.apply(lambda row: row['Bet_Y'] if row['y_win_actual']==1 else -100, axis=1)
# print(trail_df.index)
trail_df.drop(['index'], axis=1, inplace=True)
trail_df.reset_index(drop=True, inplace=True)
trail_df.reset_index(inplace=True)
print(trail_df.index)
trail_df.to_excel(file_path, index=False)

# %%
# trail_df.reset_index(drop=True, inplace=True)
# print(trail_df)


# %%
