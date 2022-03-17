import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from modules.utils import plotCorrelationMatrix

covid3 = pd.read_csv('./data/AH_Monthly_Provisional_Counts_of_Deaths_for_Select_Causes_of_Death_by_Sex__Age__and_Race_and_Hispanic_Origin.csv')
covid4 = pd.read_csv('./data/AH_Provisional_COVID-19_Deaths_and_Contributing_Conditions_by_Sex__Race_and_Hispanic_Origin__and_Age__2020.csv')

def init_data_1st_set():
  covid1 = pd.read_csv('./data/Conditions_Contributing_to_COVID-19_Deaths__by_State_and_Age__Provisional_2019.csv')
  covid2 = pd.read_csv('./data/Conditions_Contributing_to_COVID-19_Deaths__by_State_and_Age__Provisional_2020-2022.csv')
  df_all = covid1.append(covid2)
  return df_all

def pre_process_data_set1(df):
  df['Year'] = df["Year"].apply(lambda x: 2022 if math.isnan(x) else int(x))
  df = df.dropna(subset = ["Month"])
  df_fil = df[["Year", "Month", "Condition","Age Group", "COVID-19 Deaths"]]
  df_fil["Month"] = df_fil["Month"].apply(lambda x: int(x)) 
  df_fil["date"] = df_fil['Month'].map(str)+ '-' +df_fil['Year'].map(str)
  df_fil['date'] = pd.to_datetime(df_fil['date'], format='%m-%Y')
  return df_fil

def year_graph_total_death_set1(df):
  datedf = df.groupby("date").sum()["COVID-19 Deaths"]
  datedf.plot()
  plt.title("Total Deaths By Covid in the US")
  plt.xlabel("Time")
  plt.ylabel("Number of Deaths (in 100,000)")
  plt.savefig("result/Total Deaths By Covid in the US from 2020 to 2022", bbox_inches='tight', pad_inches=0.2)
  plt.clf()
  
def run_covid_analysis():
  df = init_data_1st_set()
  df = pre_process_data_set1(df)
  year_graph_total_death_set1(df)
  plotCorrelationMatrix(df, 10, "Covid Set 1 Dataset")