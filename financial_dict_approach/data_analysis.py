import pandas as pd
from sklearn.utils import shuffle
import numpy as np

def pos_neg_neu(x):
    if x == 'positive':
        return 1
    elif x == 'negative':
        return -1
    else:
        return 0

data = pd.read_csv('balanced_dataset_sentiment.csv')
print(data.columns)

#split data
data_1 = data[data['is_volatile'] == 1]
data_0 = data[data['is_volatile'] == 0]

#find correlations
features = ['positive', 'negative', 'litigious', 'constraining', 'uncertainty', 'strong_modal', 'moderate_modal', 'weak_modal']
print("volatility:        0         |        1")
for feature in features:
    vol = data_1[feature].mean()
    not_vol = data_0[feature].mean()
    print("{}: {} | {}".format(feature, not_vol, vol))