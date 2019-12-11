import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def clean_raw():

  """
  Assumption: data saved in "credit_card_data.csv"
  :return: cleaned dataset to be passed to further data transformation.

  # we drop small numbers of data points, calculate payment as % of prev month,
  and group small sample size groups together
  """
  df = pd.read_csv('credit_card_data.csv', skiprows = 1)
  df.set_index('ID', inplace=True)

  df.columns = df.columns.str.lower()
  df.rename({'default payment next month': 'default'}, axis= 1, inplace= True)

  # drop the education = 0 and marriage = 0 they are likely unknowns
  # we're losing 14 + 54 data points
  df.drop(df[df['education']==0].index, inplace=True)
  df.drop(df[df['marriage']==0].index, inplace=True)

  # calculate payment as a % of prev months balance
  for i in range(1,6):
    df[f'pay%_{i}'] = df[f'pay_amt{i}'] / df[f'bill_amt{i+1}']

  df['arrears_1'] = df['pay_0'].map(lambda x: 3 if x >= 3 else x)

  # pay_i: group 3 or more arrears into a new series, due to low data in 3m+ bins
  for i in range(2,7):
    df[f'arrears_{i}'] = df[f'pay_{i}'].map(lambda x: 3 if x >= 3 else x)

  # education: group 4,5,6 into one bin as these are all unkowns

  df['education'] = df['education'].map(lambda x: 4 if x >= 4 else x)
  df.education.value_counts()

  return df
