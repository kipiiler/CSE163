from asyncio.windows_events import NULL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_data_validation as tfdv
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from utils import plotScatterMatrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def import_data():
  df1 = pd.read_csv("./data/alzheimer.csv")
  df2 = pd.read_csv("./data/oasis_cross-sectional.csv")
  df3 = pd.read_csv("./data/clinical-data-for Alzheimers.csv")
  df4 = pd.read_csv("./data/clinical-data-for Alzheimers.csv")
  df34 = df3.append(df4)
  df34['Gender'] = df34['Gender'].apply(lambda x: "F" if x == "female" else "M")
  df34 = df34.rename(columns = {"Gender":"M/F", "mmse": "MMSE", "ageAtEntry": "Age", "cdr":"CDR" })
  df34fil = df34[["M/F", "MMSE", "Age", "CDR"]]
  df34fil["CDR"] = df34fil["CDR"].apply(lambda x: x/3 if x > 1 else int(x))
  df1 = df1.rename(columns={"EDUC":"Educ"})
  df21 = df2.append(df1)
  df_total = df21.append(df34fil)
  return df_total


def fillna_data(df_total):
  df_total['MMSE'] = df_total['MMSE'].apply(lambda x: np.NaN if x == "?" else x)
  for i in range(3,11):
    df_total[df_total.columns[i]] = pd.to_numeric(df_total[df_total.columns[i]])
    df_total[df_total.columns[i]] = df_total[df_total.columns[i]].fillna(df_total[df_total.columns[i]].mean())
  return df_total


def process_data(df_total):
  df_total = df_total.drop("Delay", axis=1)
  df_total = df_total.drop("Group", axis=1)
  df_total = df_total.drop("ID", axis=1)
  df_total = df_total.drop("Hand", axis=1)
  df_total = pd.get_dummies(df_total)
  return df_total

def plot_accuracies(accuracies,row, column, name):

    sns.relplot(kind='line', x=row, y=column, data=accuracies)
    plt.title(f'{name} {column} as {row} Changes')
    plt.xlabel(f'{row}')
    plt.ylabel(f'{column}')

    plt.savefig(f'result/ml/{name} {column} as {row} Changes')

df = import_data()
df = fillna_data(df)
df = process_data(df)

plotScatterMatrix(df, 20, 7, "ML")

labels = df["CDR"]
feats = df.drop("CDR", axis=1)

train_stats = tfdv.generate_statistics_from_dataframe(feats)
tfdv.visualize_statistics(train_stats)

min_max_scaler = MinMaxScaler()

column_names_to_normalize = ['Age', 'Educ', 'SES', 'MMSE', 'eTIV',"nWBV", "ASF"]
x = feats[column_names_to_normalize].values
feats_scaled = min_max_scaler.fit_transform(x)
normalized_features = pd.DataFrame(feats_scaled, columns=column_names_to_normalize, index = feats.index)
feats[column_names_to_normalize] = normalized_features
feats

standard_scaler = StandardScaler()

column_names_to_standardize = ['Age']
x = feats[column_names_to_standardize].values
feats_scaled = standard_scaler.fit_transform(x)

standardized_features = pd.DataFrame(feats_scaled, columns=column_names_to_standardize, index = feats.index)
feats[column_names_to_standardize] = standardized_features
feats

best_score = 99999
best_parameters = 1
depth_score = 1
best_leaf = 2

accuracies = []
for i in range(1,5):
  for j in range(1,100):
    for k in range(2,100):
      features_train, features_test, labels_train, labels_test = train_test_split(feats, labels, test_size = i/10)
      model = DecisionTreeRegressor(max_depth=j, max_leaf_nodes=k)
      model.fit(features_train, labels_train)

      train_pred = model.predict(features_train)
      test_pred = model.predict(features_test)

      error = mean_squared_error(test_pred, labels_test)
      error_train = mean_squared_error(train_pred, labels_train)
      score = error
      if score < best_score:
            best_score = score
            best_parameters = i
            depth_score = j
            best_leaf = k
      accuracies.append({'max depth': j, 'split': i, 'MSE': error, "TMSE": error_train,"Leaf": k})

accuracies = pd.DataFrame(accuracies)
    
print("Best score:" + str(best_score))
print("Best depth:" + str(depth_score))
print("Best max leaf:" + str(best_leaf))
print("Best size:" + str(best_parameters/10))


plot_accuracies(accuracies=accuracies,row="split", column="MSE", name="test_valid")
plot_accuracies(accuracies=accuracies,row="max depth", column="MSE", name="test_valid")
plot_accuracies(accuracies=accuracies,row="Leaf", column="MSE", name="test_valid")

plot_accuracies(accuracies=accuracies,row="split", column="TMSE", name="train_valid")
plot_accuracies(accuracies=accuracies,row="max depth", column="TMSE", name="train_valid")
plot_accuracies(accuracies=accuracies,row="Leaf", column="TMSE", name="train_valid")

