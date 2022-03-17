import dis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import geopandas as gpd

from modules.utils import get_states, plotCorrelationMatrix

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

def set1_alzheimer_process(df):
  aiz = df[df["Condition"] == "Alzheimer disease"]
  aizgroup = aiz.groupby("date").sum()["COVID-19 Deaths"]
  aizgroup = aizgroup.to_frame()
  aizgroup = aizgroup.rename(columns={"COVID-19 Deaths": "COVID-19 Deaths due to Alzermeir"})
  datedf = df.groupby("date").sum()["COVID-19 Deaths"]
  df_final = datedf.to_frame().join(aizgroup, on="date")
  return df_final

def year_graph_comparision_set_1(df):
  df.plot()
  plt.title("Comparision of total COVID deaths by and not by Alzermeir in the US")
  plt.xlabel("Time")
  plt.ylabel("Number of Deaths (in 100,000)")
  plt.savefig("result/Comparision of total COVID deaths by and not by Alzermeir in the US from 2020 to 2022", bbox_inches='tight', pad_inches=0.2)
  plt.clf()

def year_graph_death_by_Alzermeir(df):
  aiz = df[df["Condition"] == "Alzheimer disease"]
  aizgroup = aiz.groupby("date").sum()["COVID-19 Deaths"]
  aizgroup = aizgroup.to_frame()
  aizgroup = aizgroup.rename(columns={"COVID-19 Deaths": "COVID-19 Deaths due to Alzermeir"})
  aizgroup.plot()
  plt.title("Total COVID deaths by Alzermeir in the US")
  plt.xlabel("Time")
  plt.savefig("result/Total COVID deaths by Alzermeir in the US from 2020 to 2022", bbox_inches='tight', pad_inches=0.2)
  plt.clf()

def graph_set_1_correlation(df_final):
  sns.lineplot(data=df_final, x="COVID-19 Deaths", y ="COVID-19 Deaths due to Alzermeir")
  plt.savefig("result/Correlation of Total COVID deaths and Total COVID deaths by Alzermeir in the US from 2020 to 2022", bbox_inches='tight', pad_inches=0.2)
  plt.clf()

def plot_year_covid_death_by_column(df, year, columns):
  df_year = df[df["Year"] == year]
  df_year = df_year.dropna()
  df_year = df_year.groupby(columns).sum().reset_index()
  if "COVID-19 Deaths" in df_year.columns:
    df_year = df_year.rename(columns={"COVID-19 Deaths":"Alzheimer"})
  g = sns.barplot(data=df_year, x=columns, y="Alzheimer")
  g.set(ylim=(10, 80000))
  plt.ylabel("COVID-19 Deaths due to Alzheimer")
  plt.setp(g.get_xticklabels(), rotation=-45)
  plt.savefig(f'result/Covid Death due to alzheimer by {columns} in {year}',bbox_inches='tight', pad_inches=0.2)
  plt.clf()

def plot_age_by_group_set_one(df):
  alz = df[df["Condition"] == "Alzheimer disease"]
  plot_year_covid_death_by_column(alz, 2020, "Age Group")
  plot_year_covid_death_by_column(alz, 2021, "Age Group")
  plot_year_covid_death_by_column(alz, 2022, "Age Group")


def init_date_2nd_set():
  covid3 = pd.read_csv('./data/AH_Monthly_Provisional_Counts_of_Deaths_for_Select_Causes_of_Death_by_Sex__Age__and_Race_and_Hispanic_Origin.csv')
  covid4 = pd.read_csv('./data/AH_Provisional_COVID-19_Deaths_and_Contributing_Conditions_by_Sex__Race_and_Hispanic_Origin__and_Age__2020.csv')
  covid3fil = covid3[["Date Of Death Year", "Sex", "Race/Ethnicity", "AgeGroup", "AllCause", "Alzheimer disease (G30)"]]
  covid3fil = covid3fil.rename(columns={"Date Of Death Year": "Year","AgeGroup": "Age Group", "Race/Ethnicity": "Race", "AllCause":"Total", "Alzheimer disease (G30)": "Alzheimer"})
  for i in range(6,19):
    covid4[covid4.columns[i]] = covid4[covid4.columns[i]].apply(lambda x: 0 if math.isnan(x) else int(x))
  covid4['Total'] = covid4.iloc[:,6:19].sum(axis=1)
  covid4fil = covid4[["Year", "Sex", "Race and Hispanic Origin", "Age Group", "Total", "Alzheimer Disease"]]
  covid4fil = covid4fil.rename(columns={"Race and Hispanic Origin": "Race", "Alzheimer Disease": "Alzheimer"})
  covid34=covid3fil.append(covid4fil)
  covid34["Sex"] = covid34["Sex"].apply( lambda x : "F" if x in ["F", "Female", "Female (F)"] else "M")
  return covid34

def plot_death_conditions(raw):
  df_fil = raw[raw["Condition"]!= "COVID-19"].groupby("Condition").sum().reset_index()
  # .plot(kind="bar",x="COVID-19 Deaths", y="Condition")
  sns.barplot(x="COVID-19 Deaths", y="Condition", data=df_fil)
  plt.title("Condition Contributing to COVID19 Deaths")
  plt.xlabel("Number of Deaths")
  plt.ylabel("Conditions")
  plt.savefig("result/Condition Contributing to COVID19 Deaths.png", bbox_inches='tight', pad_inches=0.2)
  plt.clf()

def n_largest_state_year(df_origin, year):
    df = df_origin
    df["Year"] = pd.to_numeric(df["Year"])
    df = df[df["Year"] == year]
    df = df[df["Condition"] == "Alzheimer disease"]
    df = df[df["State"] != "United States"]
    df = df.groupby("State").sum().reset_index()
    df = df.nlargest(10, "COVID-19 Deaths")
    sns.barplot(x="COVID-19 Deaths", y="State", data=df)
    plt.title(f'States with highest death of COVID19 due to Alzheimer in {year}')
    plt.xlabel("States")
    plt.ylabel("Number of Deaths")
    plt.savefig(f'result/States with highest death of COVID19 due to Alzheimer in {year}.png', bbox_inches='tight', pad_inches=0.2)
    plt.clf()

def plot_n_largest_state_year(raw):
    n_largest_state_year(raw,2020)
    n_largest_state_year(raw,2021)
    n_largest_state_year(raw,2022)

def distribution_year(df_origin, year):
  df = df_origin
  df["Year"] = pd.to_numeric(df["Year"])
  df = df[df["Year"] == year]
  df["NAME"] = df["State"]
  df = df[(df['NAME'] != 'Alaska') & (df['NAME'] != 'Hawaii')]
  df = df[df["Condition"] == "Alzheimer disease"]
  df = df.groupby("NAME").sum().reset_index()
  state  = get_states()
  df_final = state.merge(df, on ="NAME")
  fig, ax = plt.subplots(1,1, figsize=(10,10))
  state= state[(state['NAME'] != 'Alaska') & (state['NAME'] != 'Hawaii')]
  state.plot(ax=ax, color="#EEEEEE")
  df_final.plot("COVID-19 Deaths", cmap='Reds',legend=True, ax=ax)
  plt.title(f'Heatmap of COVID19 deaths due to Alzheimer of US in {year}')
  plt.savefig(f'result/Heatmap of COVID19 deaths due to Alzheimer of US in {year}.png', bbox_inches='tight', pad_inches=0.2)
  plt.clf()

def plot_annual_distribution(raw):
  distribution_year(raw, 2020)
  distribution_year(raw, 2021)
  distribution_year(raw, 2022)


def plot_sex_diffrences(raw_2):
  df_s = raw_2.groupby("Sex").sum().reset_index()
  figq, axq = plt.subplots(1,1,figsize=(10,10))
  axq = sns.barplot(data=df_s, x="Sex", y="Total")
  plt.title("COVID deaths due to Alzheimer breakdown by Sex")
  plt.ylabel("Number of Deaths")
  plt.savefig('result/COVID deaths due to Alzheimer breakdown by Sex')
  plt.clf()


def plot_race_diffrences(raw_2):
  df_r = raw_2.groupby("Race").sum().reset_index()
  figq, axq = plt.subplots(1,1,figsize=(10,10))
  axq = sns.barplot(data=df_r, x="Total", y="Race")
  plt.title("COVID deaths due to Alzheimer breakdown by Race")
  plt.ylabel("Number of Deaths")
  plt.savefig('result/COVID deaths due to Alzheimer breakdown by Race')
  plt.clf()

def correlation_run(df1,df2,total):
  plotCorrelationMatrix(df1, 10, "Covid Set 1 Dataset")
  plt.clf()
  plotCorrelationMatrix(df2, 10, "Covid Set 2 Dataset")
  plt.clf()
  plotCorrelationMatrix(total, 10, "Covid Total Dataset")
  plt.clf()

def set_2_pre_process(raw_2):
  df_p = raw_2.groupby("Year").sum().reset_index()
  df_p["Year"] = df_p["Year"].apply(lambda x: str(x))
  return df_p

def transform_set_1(df_final):
  df_t = df_final.reset_index()
  df_t["date"] = df_t["date"].apply(lambda x: str(x)[0:4])
  df_t = df_t.groupby("date").sum()[["COVID-19 Deaths","COVID-19 Deaths due to Alzermeir"]].reset_index()
  df_t = df_t.rename(columns = {"date":"Year", "COVID-19 Deaths": "Total", "COVID-19 Deaths due to Alzermeir":"Alzheimer" })
  return df_t

def plot_set_total_death(df, name):
  fig1, ax1 = plt.subplots(figsize=(10, 10))
  sns.lineplot(data=df, ax=ax1, x="Year", y="Total", label="Total")
  sns.lineplot(data=df, ax=ax1, x="Year", y="Alzheimer", label="Alzheimer")
  plt.title(f'Covid-19 Death Total and due to Alzheimer ({name})')
  plt.ylabel("Number of Deaths")
  plt.savefig(f'result/Covid-19 Death Total and due to Alzheimer ({name}).png')
  plt.clf()

def plot_set_alzheimer(df, name):
  sns.lineplot(data=df, x="Year", y="Alzheimer", label="Alzheimer")
  plt.title(f'Covid-19 Death due to Alzheimer ({name})')
  plt.ylabel("Number of Deaths")
  plt.savefig(f'result/Covid-19 Deaths due to Alzheimer ({name}) only.png')
  plt.clf()

def plot_set_correlation(df, name):
  sns.relplot(data=df, x="Total", y="Alzheimer", kind='line')
  plt.title(f'Correlation of COVID-19 Deaths and Alzheimer Deaths ({name})')
  plt.savefig(f'result/Correlation of COVID-19 Deaths and Alzheimer Deaths ({name}).png')
  plt.clf()

def plot_set(df, name):
  plot_set_total_death(df, name)
  plot_set_alzheimer(df, name)
  plot_set_correlation(df, name)

def merge_two_set(set_1,set_2):
  return set_1.append(set_2).groupby("Year").sum().reset_index()


def run_covid_analysis():
  raw_1 = init_data_1st_set()
  plot_n_largest_state_year(raw_1)
  plot_death_conditions(raw_1)
  plot_annual_distribution(raw_1)
  df = pre_process_data_set1(raw_1)
  year_graph_total_death_set1(df)
  df_final = set1_alzheimer_process(df)
  year_graph_comparision_set_1(df_final)
  year_graph_death_by_Alzermeir(df)
  graph_set_1_correlation(df_final)
  plot_age_by_group_set_one(df)

  raw_2 = init_date_2nd_set()
  plot_race_diffrences(raw_2)
  plot_race_diffrences(raw_2)
  df_2 = set_2_pre_process(raw_2)
  plot_set(df_2,"Set 2")
  df_1_t = transform_set_1(df_final)
  df_all = merge_two_set(df_1_t, df_2)
  plot_set(df_all, "Total")
  correlation_run(df_1_t,df_2,df_all)
