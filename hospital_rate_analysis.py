import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

hospitalRate = pd.read_csv("data/Hospitalization_Rate_Related_To_Alzheimer_s_Or_Other_Dementias_2008-2017.csv")

def read_rate():
  return pd.read_csv("data/Hospitalization_Rate_Related_To_Alzheimer_s_Or_Other_Dementias_2008-2017.csv")

hospitalRate.groupby("Year").sum().reset_index().plot(x="Year", y="Value")

hospitalRate.groupby("Race/ ethnicity").sum().reset_index().plot(kind="bar",x="Race/ ethnicity", y="Value")

hospitalRate.groupby("Jurisdiction").sum().reset_index().nlargest(10,"Value").plot(kind="bar",x="Jurisdiction", y="Value")  