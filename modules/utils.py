import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import zipfile
import os
import geopandas as gpd
my_path = os.path.abspath(__file__) 
from constants import MY_PATH_TO_RESULT


def save_file(url, file_name):
  r = requests.get(url)
  with open(file_name, 'wb') as f:
    f.write(r.content)

def get_countries_geodata():
    save_file('https://courses.cs.washington.edu/courses/cse163/19sp/' +
          'files/lectures/05-13/data.zip', 'data.zip')

    with zipfile.ZipFile("data.zip","r") as zip_ref:
        zip_ref.extractall()

    countries = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    return countries

def get_WHO_countries_code():
   country_codes = pd.read_csv("data/country_codes.csv")
   return country_codes

def get_states():
    states = gpd.read_file("data/cb_2014_us_state_20m.shp")
    return states

def get_default_geodata():
   world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
   return world


def plotScatterMatrix(df, plotSize, textSize, name):
    df = df.select_dtypes(include=[np.number])  # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    # keep columns where there are more than 1 unique values
    df = df[[col for col in df if df[col].nunique() > 1]]
    columnNames = list(df)
    if len(columnNames) > 10:  # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[
                                    plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2),
                          xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.savefig(
        f'{MY_PATH_TO_RESULT}/Scatter and Density Plot for {name}.png', bbox_inches='tight', pad_inches=0.2)


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth, filename):
    df = df.dropna('columns')  # drop columns with NaN
    # keep columns where there are more than 1 unique values
    df = df[[col for col in df if df[col].nunique() > 1]]
    if df.shape[1] < 2:
        print(
            f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth),
               dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.savefig(
        f'{MY_PATH_TO_RESULT}/Correlation Matrix for {filename}.png', pad_inches=0.2)
