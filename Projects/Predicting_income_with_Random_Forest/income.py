def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

income_data = pd.read_csv('income.csv', header = 0, delimiter = ", ")
print(income_data.iloc[0])
print(income_data.columns)

income = income_data[['income']]

data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex' ]]

train_data, train_label, test_data, test_label = train_data_split(data, income, train_size = 0.8, test_size = 0.2, random_state = 100)







