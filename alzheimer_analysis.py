import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import geopandas as gpd
from constants import DEATH_TABLES

from modules.utils import get_WHO_countries_code, get_countries_geodata, get_default_geodata

def modify_country_code():
  country_code = get_WHO_countries_code()
  country = country_code.rename(columns={"country":"Country"})
  return country

def mort_preprocessing(mort_set):
  country = modify_country_code()
  country = country.rename(columns={"name":"country"})
  merge_set = mort_set.merge(country, on="Country")
  filter_columns = ["Year","Cause","Sex", "country"]
  filter_columns.extend(list(merge_set.columns[9:35]))
  merge_set = merge_set[filter_columns]
  result = merge_set[merge_set["Cause"].str.contains("G30")]
  result = result.drop("Cause", axis=1)
  result = result.rename(columns = DEATH_TABLES)
  return result

def n_most_death(df, n):
  list_of_value = list(df.columns)
  list_of_value.remove("country")
  df_fil = df.groupby("country").sum()[list_of_value].reset_index()
  for column in df_fil.columns[3:28]:
    fig, ax = plt.subplots(1,1, figsize=(15, 5))
    sns.barplot(x="country", y=column, ax=ax, data=df_fil.nlargest(n, column))
    plt.title(f'Top {str(n)} countries Alzeimer {column}')
    plt.xlabel("Countries")
    plt.ylabel('Numbers of Deaths')
    plt.savefig(f'result/top5/Top {str(n)} countries Alzeimer {column}.png')
    plt.clf()


def plot_rate_annually(df, country=None):
  lis = list(df.columns)
  lis.remove("Year")
  lis.remove('country')
  if country:
    df = df[df['country'] == country].groupby("Year").sum()[lis].reset_index()
  else:
    df = df.groupby("Year").sum()[lis].reset_index()
  figtest, axtest = plt.subplots(1,1,figsize=(10,10))
  for i in df.columns[2:27]:
    sns.lineplot( ax=axtest, x="Year", y=i, data=df, label=i)
  axtest.legend(bbox_to_anchor=(1, 1))
  label = "the worlds" if country == None else country
  plt.title(f'Number of Death in {label} overtime')
  plt.savefig(f'result/Number of Death in {label} overtime.png')
  plt.clf()


def read_all_mort_set():
  mort = []
  for i in range(1,5):
    mort.append(pd.read_csv("data/Morticd10_part"+str(i)+".csv"))
  result = pd.concat(mort)
  print("Date size before : "+ str(result.shape))
  result = mort_preprocessing(result)
  print("Date size after preprocess : "+ str(result.shape))
  return result

def geo_map_dementia_death(df, continent=None):
  world = get_default_geodata()
  df_clone = df    
  df_clone = df_clone.groupby('country').sum().reset_index()
  df_clone["name"] = df_clone["country"]
  df_merged = world.merge(df_clone, on="name")
  figa, axa = plt.subplots(1,1, figsize=(20,10))
  if continent:
    df_merged = df_merged[df_merged["continent"] == continent]
    world = world[world["continent"] == continent]
  world.plot(ax=axa, color='#EEEEEE')
  df_merged.plot(column="Deaths at all ages",cmap="flare", ax=axa, legend=True)
  label = "the worlds" if continent == None else continent
  plt.title(f'Distribution of Death in {label} overworld')
  plt.savefig(f'result/Distribution of Death in {label}.png')
  plt.clf()

def country_stats_transform(df):
  df_clone = df
  df_clone = df_clone[df_clone["Year"] == 2016].groupby('country').sum().reset_index()
  df_clone["NAME_EN"] = df_clone["country"]
  countries = get_countries_geodata()
  df_merged = countries.merge(df_clone, on="NAME_EN")
  df_merged["Death Rate"] = df_merged["Deaths at all ages"]/df_merged["POP_EST"] 
  return df_merged

def correlation_plot(df, y):
  fig, [[ax1,ax2,ax3],[ax4,ax5,ax6]] = (plt.subplots(2,3, figsize=(20,10)))
  sns.regplot( x="POP_EST", ax=ax1, y=y,data=df)
  sns.regplot( x="LABELRANK", ax=ax2, y=y,data=df)
  sns.regplot( x="POP_RANK", ax=ax3, y=y,data=df)
  sns.regplot( x="GDP_MD_EST", ax=ax4, y=y,data=df)
  sns.scatterplot( x="ECONOMY", ax=ax5, y=y,data=df)
  sns.scatterplot( x="INCOME_GRP", ax=ax6, y=y,data=df)

  ax1.set_title(f'Population Estimate vs {y}')
  ax2.set_title(f'Country Rank vs {y}')
  ax3.set_title(f'Population Rank vs {y}')
  ax4.set_title(f'GDP vs {y}')
  ax5.set_title(f'Economy type vs {y}')
  ax6.set_title(f'Income group vs {y}')

  plt.setp(ax1, xlabel='Population Estimate')
  plt.setp(ax2, xlabel='Country Ranking')
  plt.setp(ax3, xlabel='Population Rank')
  plt.setp(ax4, xlabel='GDP')
  plt.setp(ax5, xlabel='Economy type')
  plt.setp(ax6, xlabel='Income group')
  plt.setp([ax1, ax2, ax3, ax4, ax5], ylabel='Number of deaths due to alzheimer')
  plt.setp(ax5.get_xticklabels(), rotation=-45)
  plt.setp(ax6.get_xticklabels(), rotation=-45)

  plt.savefig(f'result/Correllation of Alzheimer {y} to some of country stats.png')
  plt.clf()


def run_alzheimer_analysis():
  print("RUNNING ALZHEIMER_ANALYSIS")
  all_mort = read_all_mort_set()
  n_most_death(all_mort, 5)
  plot_rate_annually(all_mort)
  plot_rate_annually(all_mort, "United States of America")
  plot_rate_annually(all_mort, "Turkey")
  geo_map_dementia_death(all_mort)
  geo_map_dementia_death(all_mort, "North America")
  all_mort_transformed = country_stats_transform(all_mort)
  correlation_plot(all_mort_transformed, "Deaths at all ages")
  correlation_plot(all_mort_transformed, "Death Rate")
  print("ALZHEIMER_ANALYSIS DONE")