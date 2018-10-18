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

data = pd.read_csv('news_to_emotions_relative_test.csv')

#split data
data_0 = data[data['is_volatile'] == 0]
data_1 = data[data['is_volatile'] == 1]

#find correlations
features = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'sentiment_dict', 'sentiment_lib_avg' ]
print("volatility:     0         |       1")
for feature in features:
    neg = data_0[feature].mean()
    pos = data_1[feature].mean()
    print("{}: {} | {}".format(feature, neg, pos))